"""Run validation and persist accepted records, rejected rows, and audit report."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.ingestion.load import load_raw
from src.utils.io import write_dataset, write_json
from src.validation.rules import validate


def validate_and_save(raw: Path, accepted_root: Path, error_root: Path,
                      report_root: Path, parquet: bool = True, now: str = '2026-11-12T00:00:00Z') -> dict:
    result = validate(load_raw(raw), now=now)
    for name in ('users', 'items', 'interactions'):
        write_dataset(result.accepted[name], Path(accepted_root) / name, parquet=parquet)
        # Quarantine files are always CSV to make them easy to inspect.
        write_dataset(result.rejected[name], Path(error_root) / f'{name}_rejected', parquet=False)
    write_json(Path(report_root) / 'validation_report.json', result.report)
    return result.report
