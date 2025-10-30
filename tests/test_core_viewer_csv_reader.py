import os
import io
import unittest
import tempfile
from pathlib import Path

import pandas as pd

from coreviewercsvreader.core_viewer_csv_reader import (
    _drop_empty_last_column,
    _get_line_containing,
    _fix_csv_file,
    read_coresensing_csv,
    read_many,
    to_parquet,
)


class TestDropEmptyLastColumn(unittest.TestCase):
    def test_drop_when_last_all_na(self):
        # last col all NaN -> drop
        df = pd.DataFrame({"a": [1, 2], "b": [None, None]})
        out = _drop_empty_last_column(df)
        self.assertListEqual(list(out.columns), ["a"])  # dropped

    def test_drop_when_last_all_empty_strings(self):
        # last col all empty strings -> drop
        df = pd.DataFrame({"a": [1], "b": [""]})
        out = _drop_empty_last_column(df)
        self.assertListEqual(list(out.columns), ["a"])  # dropped

    def test_keep_when_last_has_value(self):
        # last col has value -> keep
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        out = _drop_empty_last_column(df)
        self.assertListEqual(list(out.columns), ["a", "b"])  # kept

    def test_empty_df(self):
        # no columns -> unchanged
        df = pd.DataFrame()
        out = _drop_empty_last_column(df)
        self.assertEqual(out.shape, (0, 0))


class TestGetLineContaining(unittest.TestCase):
    def test_find_single_needle(self):
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            p = Path(f.name)
            f.write("alpha\n.beta\nSensordata here\n")
        try:
            idx = _get_line_containing(p, "Sensordata")
            self.assertEqual(idx, 2)
        finally:
            p.unlink(missing_ok=True)

    def test_find_any_of_multiple_needles(self):
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            p = Path(f.name)
            f.write("foo\nbar\nbaz\n")
        try:
            idx = _get_line_containing(p, ["nope", "bar"])  # any match
            self.assertEqual(idx, 1)
        finally:
            p.unlink(missing_ok=True)

    def test_not_found_returns_minus_one(self):
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            p = Path(f.name)
            f.write("foo\nbar\n")
        try:
            self.assertEqual(_get_line_containing(p, "zzz"), -1)
        finally:
            p.unlink(missing_ok=True)


class TestFixCsvFile(unittest.TestCase):
    def test_adds_comma_to_description_line(self):
        # build a small file with Sensordata marker and description line
        content = "Header\nSensordata\nDescription without comma\n1,2,3\n"
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            p = Path(f.name)
            f.write(content)
        try:
            changed = _fix_csv_file(p)
            self.assertTrue(changed)
            # verify the description line ends with a comma now
            txt = p.read_text()
            self.assertIn("Description without comma,\n", txt)
            # idempotent second run
            changed2 = _fix_csv_file(p)
            self.assertFalse(changed2)
        finally:
            p.unlink(missing_ok=True)

    def test_returns_false_when_marker_missing(self):
        content = "no marker here\n"
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            p = Path(f.name)
            f.write(content)
        try:
            changed = _fix_csv_file(p)
            self.assertFalse(changed)
            self.assertEqual(p.read_text(), content)
        finally:
            p.unlink(missing_ok=True)


class TestPublicApis(unittest.TestCase):
    def setUp(self):
        # sample CSV from repo
        base = Path(__file__).parent / "testdata"
        self.sample_csv = base / "1-01107_2025-10-24_16-14-56.csv"
        self.sample_csv_eu = base / "1-01107_2025-10-24_16-14-56_eu.csv"
        self.assertTrue(self.sample_csv.exists(), f"Missing test file: {self.sample_csv}")
        self.assertTrue(self.sample_csv_eu.exists(), f"Missing test file: {self.sample_csv_eu}")

    def test_read_coresensing_csv_basic(self):
        df = read_coresensing_csv(self.sample_csv)
        self.assertGreaterEqual(len(df), 1)  # has rows
        self.assertGreaterEqual(df.shape[1], 1)  # has cols

    def test_read_coresensing_csv_eu_separator(self):
        # ensure EU file with ';' is parsed and header detected
        df = read_coresensing_csv(self.sample_csv_eu)
        self.assertIn("Time in ms", df.columns)
        # self.assertGreaterEqual(len(df), 1)
        # self.assertGreaterEqual(df.shape[1], 2)

    def test_read_many_concat_and_source(self):
        df = read_many([self.sample_csv, self.sample_csv], add_source=True)
        self.assertIn("__source__", df.columns)
        # two sources, but same path value
        self.assertEqual(df["__source__"].nunique(), 1)
        # concatenated rows
        self.assertGreaterEqual(len(df), 2)

    def test_read_many_empty_input(self):
        df = read_many([])
        self.assertEqual(df.shape, (0, 0))

    def test_to_parquet_roundtrip_if_engine_available(self):
        # skip if no parquet engine available
        engine_available = False
        try:
            import pyarrow  # noqa: F401
            engine_available = True
        except Exception:
            try:
                import fastparquet  # noqa: F401
                engine_available = True
            except Exception:
                engine_available = False

        if not engine_available:
            self.skipTest("No parquet engine installed")

        df = read_coresensing_csv(self.sample_csv)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.parquet"
            to_parquet(df, out)
            self.assertTrue(out.exists())
            # ensure it can be read back
            back = pd.read_parquet(out)
            self.assertEqual(len(back), len(df))
            self.assertEqual(back.shape[1], df.shape[1])


if __name__ == "__main__":
    unittest.main()
