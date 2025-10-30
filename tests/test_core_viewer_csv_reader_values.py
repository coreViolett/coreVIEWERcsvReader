# python
from pathlib import Path
import pandas as pd
import pytest

from src.coreviewercsvreader.core_viewer_csv_reader import read_coresensing_csv


def _data_path(name: str = "1-01107_2025-10-24_16-14-56_eu.csv") -> Path:
    return Path(__file__).parent / "testdata" / name


def test_all_columns_are_numeric() -> None:
    df = read_coresensing_csv(_data_path())
    for c in df.columns:
        assert pd.api.types.is_numeric_dtype(df[c]), f"Column {c} not numeric"

    df = read_coresensing_csv(_data_path(name="1-01107_2025-10-24_16-14-56.csv"))
    for c in df.columns:
        assert pd.api.types.is_numeric_dtype(df[c]), f"Column {c} not numeric"


def test_read_eu_csv_parses_values() -> None:
    df = read_coresensing_csv(_data_path())

    assert df.shape[1] == 5
    assert len(df) >= 5

    # time column (ms)
    assert int(df.iloc[0, 0]) == 0
    assert int(df.iloc[1, 0]) == 100
    assert int(df.iloc[2, 0]) == 200
    assert int(df.iloc[3, 0]) == 300
    assert int(df.iloc[4, 0]) == 400

    # first data row (EU decimals)
    assert df.iloc[0, 1] == pytest.approx(0.1071704523610606, rel=0, abs=1e-15)
    assert df.iloc[0, 2] == pytest.approx(7.666686937917348, rel=0, abs=1e-15)
    assert df.iloc[0, 3] == pytest.approx(5.297541374054617, rel=0, abs=1e-15)
    assert df.iloc[0, 4] == pytest.approx(4.389669416009838, rel=0, abs=1e-15)

    # fifth data row (index 4)
    assert df.iloc[4, 1] == pytest.approx(0.11041608348114096, rel=0, abs=1e-15)
    assert df.iloc[4, 2] == pytest.approx(7.670460343124432, rel=0, abs=1e-15)
    assert df.iloc[4, 3] == pytest.approx(5.297717298750285, rel=0, abs=1e-15)
    assert df.iloc[4, 4] == pytest.approx(4.390976285177658, rel=0, abs=1e-15)


def test_read_csv_parses_values() -> None:
    df = read_coresensing_csv(_data_path(name="1-01107_2025-10-24_16-14-56.csv"))

    assert df.shape[1] == 5
    assert len(df) >= 5

    # time column (ms)
    assert int(df.iloc[0, 0]) == 0
    assert int(df.iloc[1, 0]) == 100
    assert int(df.iloc[2, 0]) == 200
    assert int(df.iloc[3, 0]) == 300
    assert int(df.iloc[4, 0]) == 400

    # first data row
    assert df.iloc[0, 1] == pytest.approx(0.1071704523610606, rel=0, abs=1e-15)
    assert df.iloc[0, 2] == pytest.approx(7.666686937917348, rel=0, abs=1e-15)
    assert df.iloc[0, 3] == pytest.approx(5.297541374054617, rel=0, abs=1e-15)
    assert df.iloc[0, 4] == pytest.approx(4.389669416009838, rel=0, abs=1e-15)

    # fifth data row (index 4)
    assert df.iloc[4, 1] == pytest.approx(0.11041608348114096, rel=0, abs=1e-15)
    assert df.iloc[4, 2] == pytest.approx(7.670460343124432, rel=0, abs=1e-15)
    assert df.iloc[4, 3] == pytest.approx(5.297717298750285, rel=0, abs=1e-15)
    assert df.iloc[4, 4] == pytest.approx(4.390976285177658, rel=0, abs=1e-15)
