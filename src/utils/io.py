"""Filesystem utilities: canonical CSV and optional Parquet output."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')


def write_dataset(frame: pd.DataFrame, output: Path, parquet: bool = True) -> dict:
    """Always write CSV; write Parquet when pyarrow is installed.

    Failure to write Parquet is not masked: when parquet=True it raises and
    the caller is expected to install requirements.txt.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix('.csv')
    frame.to_csv(csv_path, index=False)
    saved = {'csv': str(csv_path)}
    if parquet:
        parquet_path = output.with_suffix('.parquet')
        frame.to_parquet(parquet_path, index=False, engine='pyarrow')
        saved['parquet'] = str(parquet_path)
    return saved
