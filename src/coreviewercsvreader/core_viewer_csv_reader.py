from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence, Union, List, Optional
import os
import tempfile
import pandas as pd

__all__ = ["read_coresensing_csv", "read_many", "to_parquet"]


def _drop_empty_last_column(df: pd.DataFrame) -> pd.DataFrame:
    # drop last column if all NaN or empty strings
    if df.shape[1] == 0:
        return df
    last = df.columns[-1]
    col = df[last]
    all_na = col.isna().all()
    all_empty = (
        col.dtype == object
        and col.fillna("").astype(str).str.strip().eq("").all()
    )
    return df.iloc[:, :-1] if (all_na or all_empty) else df


def _get_line_containing(
    path: Union[str, Path],
    needles: Union[str, Sequence[str]],
    *,
    encoding: str = "utf-8",
) -> int:
    # return first 0-based line index containing any needle, else -1
    if isinstance(needles, str):
        needles = [needles]
    needles = list(needles)

    with open(path, "r", encoding=encoding, errors="replace") as f:
        for i, line in enumerate(f):
            if any(n in line for n in needles):
                return i
    return -1


def _detect_separator(path: Union[str, Path], *, encoding: str = "utf-8") -> str:
    # detect sep from header line right after 'Sensordata'
    p = Path(path)
    marker_idx = _get_line_containing(p, "Sensordata", encoding=encoding)
    if marker_idx < 0:
        return ","
    header_idx = marker_idx + 1
    with open(p, "r", encoding=encoding, errors="replace") as f:
        for i, line in enumerate(f):
            if i == header_idx:
                # simple heuristic: prefer ';' if present, else ','
                if "Time in ms;" in line:
                    return ";"
                if "Time in ms," in line:
                    return ","
                return ","
    return ","


def _fix_csv_file(
    path: Union[str, Path],
    needles: Union[str, Sequence[str]] = "Sensordata",
    *,
    encoding: str = "utf-8",
    sep_char: str = ",",
) -> bool:
    """
    Ensure description/header line (after marker) ends with trailing separator.
    Stream via temp file to avoid loading everything into memory.
    Return True if file was modified.
    """
    p = Path(path)
    marker_idx = _get_line_containing(p, needles, encoding=encoding)
    if marker_idx < 0:
        return False
    desc_idx = marker_idx + 1

    changed = False
    # use nested with to avoid long lines
    with open(
        p,
        "r",
        encoding=encoding,
        errors="replace",
        newline="",
    ) as src:
        with tempfile.NamedTemporaryFile(
            "w", delete=False, encoding=encoding, newline=""
        ) as dst:
            tmp_name = dst.name
            for i, line in enumerate(src):
                if i == desc_idx:
                    # keep original EOL
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
                    if not body.endswith(sep_char):
                        line = body + sep_char + eol
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
    sep: Optional[str] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Robustly read coreSensing CSVs:
    - assumes a header row is present
    - detects separator from header after 'Sensordata' if not provided
    - tolerates trailing delimiter (empty last column)
    - best-effort numeric casting
    """
    path = Path(path)

    # detect sep from header
    detected_sep = _detect_separator(path, encoding=encoding)
    use_sep = detected_sep if sep is None else sep
    decimal = "," if use_sep == ";" else "."
    # fix header line to end with separator if missing
    _fix_csv_file(path, "Sensordata", encoding=encoding, sep_char=use_sep)

    marker_idx = _get_line_containing(path, "Sensordata", encoding=encoding)
    skiprows = (marker_idx + 1) if marker_idx >= 0 else None

    df = pd.read_csv(
        path,
        sep=use_sep,
        decimal=decimal,
        header=0,
        engine="python",
        skip_blank_lines=True,
        on_bad_lines="warn",
        encoding=encoding,
        skiprows=skiprows,
    )

    df = _drop_empty_last_column(df)

    # best-effort numeric conversion: convert if fully numeric, else keep
    for c in df.columns:
        try:
            converted = pd.to_numeric(df[c])
            df[c] = converted
        except Exception:
            pass

    return df


def read_many(
    paths: Iterable[str | Path],
    *,
    sep: Optional[str] = None,
    encoding: str = "utf-8",
    add_source: bool = True,
) -> pd.DataFrame:
    # read each file, optionally add __source__, then concat
    frames: List[pd.DataFrame] = []
    for p in paths:
        df = read_coresensing_csv(p, sep=sep, encoding=encoding)
        if add_source:
            df = df.copy()
            df["__source__"] = str(Path(p))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def to_parquet(df: pd.DataFrame, out_path: str | Path) -> None:
    # write dataframe to parquet
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
