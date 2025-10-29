# python
from pathlib import Path
import argparse
import pandas as pd
from coreviewercsvreader.core_viewer_csv_reader import read_coresensing_csv


def main(data_dir: str = "data") -> pd.DataFrame:
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Keine CSV‑Datei in `{data_dir}` gefunden.")
    return read_coresensing_csv(files[0])


def cli() -> None:
    parser = argparse.ArgumentParser(description="Read coreSensing CSV into a pandas DataFrame.")
    parser.add_argument("input", nargs="?", default="data", help="CSV‑Datei oder Verzeichnis mit CSVs")
    parser.add_argument("--header", action="store_true", help="CSV enthält Headerzeile")
    parser.add_argument("--encoding", default="utf-8", help="Datei‑Encoding (Default: utf‑8)")
    args = parser.parse_args()

    p = Path(args.input)
    if p.is_dir():
        files = sorted(p.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"Keine CSV‑Datei in `{p}` gefunden.")
        target = files[0]
    else:
        if not p.exists():
            raise FileNotFoundError(f"Datei `{p}` nicht gefunden.")
        target = p

    df = read_coresensing_csv(target, has_header=args.header, encoding=args.encoding)
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    cli()
