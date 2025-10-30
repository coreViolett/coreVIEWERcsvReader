import unittest
from pathlib import Path

import coreviewercsvreader as cvr


class TestSmoke(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(__file__).parent / "testdata" / "1-01107_2025-10-24_16-14-56.csv"
        self.assertTrue(self.data_path.exists(), f"Missing test CSV: {self.data_path}")

    def test_read_single_file(self):
        df = cvr.read_coresensing_csv(self.data_path)
        self.assertGreaterEqual(len(df), 1)
        self.assertGreaterEqual(df.shape[1], 1)

    def test_read_many_with_source(self):
        df = cvr.read_many([self.data_path], add_source=True)
        self.assertIn("__source__", df.columns)
        self.assertEqual(df["__source__"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()

