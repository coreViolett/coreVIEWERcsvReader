# python
from pathlib import Path
import argparse
import pandas as pd
from coreviewercsvreader.core_viewer_csv_reader import read_coresensing_csv


def main(data_dir: str = "data") -> pd.DataFrame:
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV‑file in `{data_dir}`.")
    return read_coresensing_csv(files[0])


def cli() -> None:
    parser = argparse.ArgumentParser(description="Read coreSensing CSV into a pandas DataFrame.")
    parser.add_argument("input", nargs="?", default="data", help="CSV‑File")
    parser.add_argument("--encoding", default="utf-8", help="Data‑Encoding (Default: utf‑8)")
    args = parser.parse_args()

    p = Path(args.input)
    if p.is_dir():
        files = sorted(p.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV‑file in `{p}`.")
        target = files[0]
    else:
        if not p.exists():
            raise FileNotFoundError(f"File `{p}` not found.")
        target = p

    df = read_coresensing_csv(target, encoding=args.encoding)
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    cli()
