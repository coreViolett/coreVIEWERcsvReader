from pathlib import Path

from src.coreviewercsvreader import read_coresensing_csv, read_many


def _data_path() -> Path:
    p = Path(__file__).parent / "testdata" / "1-01107_2025-10-24_16-14-56.csv"
    assert p.exists(), f"Missing test CSV: {p}"
    return p


def test_read_single_file():
    df = read_coresensing_csv(_data_path())
    assert len(df) >= 1
    assert df.shape[1] >= 1


def test_read_many_with_source():
    df = read_many([_data_path()], add_source=True)
    assert "__source__" in df.columns
    assert df["__source__"].nunique() == 1
