from pathlib import Path

import pandas as pd
import pytest

from src.coreviewercsvreader.core_viewer_csv_reader import (
    _drop_empty_last_column,
    _get_line_containing,
    _fix_csv_file,
    read_coresensing_csv,
    read_many,
    to_parquet,
)


# ---- _drop_empty_last_column ----

def test_drop_when_last_all_na():
    df = pd.DataFrame({"a": [1, 2], "b": [None, None]})
    out = _drop_empty_last_column(df)
    assert list(out.columns) == ["a"]


def test_drop_when_last_all_empty_strings():
    df = pd.DataFrame({"a": [1], "b": [""]})
    out = _drop_empty_last_column(df)
    assert list(out.columns) == ["a"]


def test_keep_when_last_has_value():
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    out = _drop_empty_last_column(df)
    assert list(out.columns) == ["a", "b"]


def test_empty_df():
    df = pd.DataFrame()
    out = _drop_empty_last_column(df)
    assert out.shape == (0, 0)


# ---- _get_line_containing ----

def test_find_single_needle(tmp_path: Path):
    p = tmp_path / "f.txt"
    p.write_text("alpha\n.beta\nSensordata here\n", encoding="utf-8")
    idx = _get_line_containing(p, "Sensordata")
    assert idx == 2


def test_find_any_of_multiple_needles(tmp_path: Path):
    p = tmp_path / "g.txt"
    p.write_text("foo\nbar\nbaz\n", encoding="utf-8")
    idx = _get_line_containing(p, ["nope", "bar"])
    assert idx == 1


def test_not_found_returns_minus_one(tmp_path: Path):
    p = tmp_path / "h.txt"
    p.write_text("foo\nbar\n", encoding="utf-8")
    assert _get_line_containing(p, "zzz") == -1


# ---- _fix_csv_file ----

def test_adds_comma_to_description_line(tmp_path: Path):
    # Build a tiny file with marker and a description line
    content = "Header\nSensordata\nDescription without comma\n1,2,3\n"
    p = tmp_path / "data.csv"
    p.write_text(content, encoding="utf-8")

    changed = _fix_csv_file(p)
    assert changed is True

    txt = p.read_text(encoding="utf-8")
    assert "Description without comma,\n" in txt

    # Idempotent second run
    changed2 = _fix_csv_file(p)
    assert changed2 is False


def test_returns_false_when_marker_missing(tmp_path: Path):
    p = tmp_path / "no_marker.csv"
    content = "no marker here\n"
    p.write_text(content, encoding="utf-8")

    changed = _fix_csv_file(p)
    assert changed is False
    assert p.read_text(encoding="utf-8") == content


# ---- public APIs ----

def _data_paths() -> tuple[Path, Path]:
    base = Path(__file__).parent / "testdata"
    csv = base / "1-01107_2025-10-24_16-14-56.csv"
    csv_eu = base / "1-01107_2025-10-24_16-14-56_eu.csv"
    assert csv.exists(), f"Missing test file: {csv}"
    assert csv_eu.exists(), f"Missing test file: {csv_eu}"
    return csv, csv_eu


def test_read_coresensing_csv_basic():
    csv, _ = _data_paths()
    df = read_coresensing_csv(csv)
    assert len(df) >= 1
    assert df.shape[1] >= 1


def test_read_coresensing_csv_eu_separator():
    _, csv_eu = _data_paths()
    df = read_coresensing_csv(csv_eu)
    assert "Time in ms" in df.columns
    assert len(df) >= 1
    assert df.shape[1] >= 2


def test_read_many_concat_and_source():
    csv, _ = _data_paths()
    df = read_many([csv, csv], add_source=True)
    assert "__source__" in df.columns
    # Two sources but same path value
    assert df["__source__"].nunique() == 1
    # Concatenated rows
    assert len(df) >= 2


def test_read_many_empty_input():
    df = read_many([])
    assert df.shape == (0, 0)


def test_to_parquet_roundtrip_if_engine_available(tmp_path: Path):
    # Skip if no parquet engine
    have_engine = True
    try:
        import pyarrow  # noqa: F401
    except Exception:
        try:
            import fastparquet  # noqa: F401
        except Exception:
            have_engine = False

    if not have_engine:
        pytest.skip("No parquet engine installed")

    csv, _ = _data_paths()
    df = read_coresensing_csv(csv)
    out = tmp_path / "out.parquet"
    to_parquet(df, out)
    assert out.exists()

    back = pd.read_parquet(out)
    assert len(back) == len(df)
    assert back.shape[1] == df.shape[1]
