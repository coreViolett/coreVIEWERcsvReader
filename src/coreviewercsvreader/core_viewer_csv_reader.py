# python
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union
import os
import tempfile
import pandas as pd

__all__ = ["read_coresensing_csv", "read_many", "to_parquet"]

DEFAULT_NAMES: Optional[List[str]] = None  # z.\ B. ["seq", "c1", "c2", "c3", "c4"]


def _drop_empty_last_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] == 0:
        return df
    last = df.columns[-1]
    col = df[last]
    all_na = col.isna().all()
    all_empty = col.dtype == object and col.fillna("").astype(str).str.strip().eq("").all()
    return df.iloc[:, :-1] if (all_na or all_empty) else df


def _get_line_containing(
    path: Union[str, Path],
    needles: Union[str, Sequence[str]],
    *,
    encoding: str = "utf-8",
) -> int:
    if isinstance(needles, str):
        needles = [needles]
    needles = list(needles)

    with open(path, "r", encoding=encoding, errors="replace") as f:
        for i, line in enumerate(f):
            if any(n in line for n in needles):
                return i
    return -1  # wichtig: -1 = nicht gefunden


def _fix_csv_file(
    path: Union[str, Path],
    needles: Union[str, Sequence[str]] = "Sensordata",
    *,
    encoding: str = "utf-8",
) -> bool:
    """
    Fügt in der Description‑Zeile (erste Zeile nach 'Sensordata') ein Komma am Zeilenende hinzu.
    Arbeitet streamend über eine temporäre Datei, ohne alles in den RAM zu laden.
    Gibt True zurück, wenn eine Änderung vorgenommen wurde.
    """
    p = Path(path)
    marker_idx = _get_line_containing(p, needles, encoding=encoding)
    if marker_idx < 0:
        return False
    desc_idx = marker_idx + 1

    changed = False
    with open(p, "r", encoding=encoding, errors="replace", newline="") as src, \
         tempfile.NamedTemporaryFile("w", delete=False, encoding=encoding, newline="") as dst:
        tmp_name = dst.name
        for i, line in enumerate(src):
            if i == desc_idx:
                # EOL erkennen und beibehalten
                if line.endswith("\r\n"):
                    eol = "\r\n"
                    body = line[:-2]
                elif line.endswith("\n"):
                    eol = "\n"
                    body = line[:-1]
                elif line.endswith("\r"):
                    eol = "\r"
                    body = line[:-1]
                else:
                    eol = ""
                    body = line
                if not body.endswith(","):
                    line = body + "," + eol
                    changed = True
            dst.write(line)

    if changed:
        os.replace(tmp_name, p)
    else:
        os.unlink(tmp_name)
    return changed


def read_coresensing_csv(
    path: str | Path,
    *,
    has_header: bool = False,
    names: Optional[List[str]] = DEFAULT_NAMES,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Liest coreSensing‑CSV robust:
    - kein Header optional,
    - toleriert finales Trennzeichen (leere letzte Spalte),
    - leitet numerische Typen bestmöglich ab.
    """
    path = Path(path)

    # Optional: Datei vor dem Lesen korrigieren (Komma in Description‑Zeile anhängen)
    _fix_csv_file(path, "Sensordata", encoding=encoding)

    header = 0 if has_header else None
    marker_idx = _get_line_containing(path, "Sensordata", encoding=encoding)
    skiprows = (marker_idx + 1) if marker_idx >= 0 else None

    df = pd.read_csv(
        path,
        sep=sep,
        header=header,
        engine="python",
        skip_blank_lines=True,
        on_bad_lines="warn",
        encoding=encoding,
        skiprows=skiprows,
    )

    df = _drop_empty_last_column(df)

    if header is None:
        if names and len(names) == df.shape[1]:
            df.columns = names
        elif df.columns.dtype != object:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    return df


def read_many(
    paths: Iterable[str | Path],
    *,
    has_header: bool = False,
    names: Optional[List[str]] = DEFAULT_NAMES,
    sep: str = ",",
    encoding: str = "utf-8",
    add_source: bool = True,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in paths:
        df = read_coresensing_csv(p, has_header=has_header, names=names, sep=sep, encoding=encoding)
        if add_source:
            df = df.copy()
            df["__source__"] = str(Path(p))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def to_parquet(df: pd.DataFrame, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
