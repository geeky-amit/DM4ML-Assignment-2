"""Auditable quality checks, row quarantine, referential integrity.

Rows with any error are removed from accepted output. Warning-free schema
checking is implemented in pandas to avoid extra platform-specific validators.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from src.ingestion.load import GENRES

REQUIRED = {
    'users': ('user_id', 'age', 'gender', 'occupation', 'zip_code'),
    'items': ('item_id', 'title', 'release_date', 'video_release_date', 'imdb_url', *GENRES),
    'interactions': ('user_id', 'item_id', 'rating', 'timestamp'),
}

@dataclass
class ValidationResult:
    accepted: dict[str, pd.DataFrame]
    rejected: dict[str, pd.DataFrame]
    report: dict


def _missing(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype('string').str.strip().eq('').fillna(True)


def _positive_int(s: pd.Series) -> pd.Series:
    cleaned = s.astype('string').str.strip()
    # IDs must be decimal positive integers, not floats like 1.0.
    numeric = pd.to_numeric(cleaned, errors='coerce')
    return (cleaned.str.fullmatch(r'[0-9]+').fillna(False) & numeric.ge(1).fillna(False))


def validate(tables: dict[str, pd.DataFrame], now: str = '2026-11-12T00:00:00Z') -> ValidationResult:
    """Validate required schema, row-level quality, duplicates and foreign keys.

    `now` is injected to make timestamp validation repeatable in tests.
    Time-related invalidity: unparseable, < 1990-01-01, or > now.
    Empty release_date, video_release_date and imdb_url are valid optional metadata.
    """
    report = {'validation_version': '1.0', 'rules': {}, 'tables': {}}
    reason_lists: dict[str, pd.Series] = {}
    for name, required in REQUIRED.items():
        if name not in tables:
            raise ValueError(f'Missing required table: {name}')
        missing_cols = set(required).difference(tables[name].columns)
        if missing_cols:
            raise ValueError(f'{name} missing required columns: {sorted(missing_cols)}')
        reason_lists[name] = pd.Series([[] for _ in range(len(tables[name]))], index=tables[name].index, dtype=object)

    def flag(name: str, mask: pd.Series, reason: str) -> None:
        mask = mask.fillna(False).astype(bool)
        current = reason_lists[name]
        for idx in current.index[mask]:
            current.at[idx] = [*current.at[idx], reason]
        report['rules'][f'{name}.{reason}'] = int(mask.sum())

    for name, frame in tables.items():
        for col in REQUIRED[name]:
            if col in ('release_date', 'video_release_date', 'imdb_url'):
                continue  # Optional metadata, not a required non-null field.
            flag(name, _missing(frame[col]), f'missing_{col}')

    for name, id_col in (('users','user_id'), ('items','item_id')):
        frame = tables[name]
        flag(name, ~_positive_int(frame[id_col]), f'invalid_{id_col}_type')
        flag(name, frame[id_col].astype('string').duplicated(keep='first'), f'duplicate_{id_col}')

    users = tables['users']
    ages = pd.to_numeric(users['age'], errors='coerce')
    flag('users', ~(ages.between(1, 120) & ages.mod(1).eq(0)).fillna(False), 'invalid_age_type_or_range')
    flag('users', ~users['gender'].isin(['M','F']), 'invalid_gender')

    items = tables['items']
    for col in GENRES:
        flag('items', ~items[col].isin(['0','1']), f'invalid_genre_{col}')
    # MovieLens release dates use dd-Mon-YYYY, e.g. 01-Jan-1995.
    release = pd.to_datetime(items['release_date'].replace('', pd.NA), format='%d-%b-%Y', errors='coerce')
    flag('items', (~_missing(items['release_date'])) & release.isna(), 'invalid_release_date')

    frame = tables['interactions']
    for col in ('user_id', 'item_id'):
        flag('interactions', ~_positive_int(frame[col]), f'invalid_{col}_type')
    ratings = pd.to_numeric(frame['rating'], errors='coerce')
    flag('interactions', ~(ratings.between(1,5) & ratings.mod(1).eq(0)).fillna(False), 'invalid_rating_type_or_range')
    timestamps = pd.to_numeric(frame['timestamp'], errors='coerce')
    as_dates = pd.to_datetime(timestamps, unit='s', errors='coerce', utc=True)
    lower = pd.Timestamp('1990-01-01', tz='UTC')
    upper = pd.Timestamp(now)
    if upper.tz is None:
        upper = upper.tz_localize('UTC')
    flag('interactions', timestamps.isna() | as_dates.isna() | as_dates.lt(lower) | as_dates.gt(upper) |
         timestamps.mod(1).ne(0), 'invalid_timestamp')
    flag('interactions', frame.duplicated(subset=['user_id','item_id'], keep='first'),
         'duplicate_user_item_interaction')

    # Foreign keys reference *accepted* entities, not merely raw entity IDs.
    valid_users = users.loc[reason_lists['users'].map(len).eq(0), 'user_id'].astype(str)
    valid_items = items.loc[reason_lists['items'].map(len).eq(0), 'item_id'].astype(str)
    flag('interactions', ~frame['user_id'].astype(str).isin(valid_users), 'unknown_user_id')
    flag('interactions', ~frame['item_id'].astype(str).isin(valid_items), 'unknown_item_id')

    accepted, rejected = {}, {}
    for name, frame in tables.items():
        errors = reason_lists[name]
        bad = errors.map(len).gt(0)
        accepted[name] = frame.loc[~bad].reset_index(drop=True).copy()
        rejected[name] = frame.loc[bad].copy()
        rejected[name]['error_reasons'] = errors.loc[bad].map(lambda x: '|'.join(x))
        rejected[name] = rejected[name].reset_index(drop=True)
        report['tables'][name] = {
            'input_rows': len(frame), 'accepted_rows': len(accepted[name]),
            'rejected_rows': len(rejected[name]),
            'acceptance_rate': round(len(accepted[name]) / len(frame), 6) if len(frame) else None,
            'rejection_examples': rejected[name].head(3).to_dict(orient='records'),
        }
    report['total_rejected_rows'] = sum(len(x) for x in rejected.values())
    report['total_input_rows'] = sum(len(x) for x in tables.values())
    report['status'] = 'PASS' if report['total_rejected_rows'] == 0 else 'PASS_WITH_QUARANTINE'
    return ValidationResult(accepted, rejected, report)
