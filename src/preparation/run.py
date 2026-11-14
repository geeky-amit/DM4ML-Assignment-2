"""Time-aware preprocessing of already validated MovieLens tables.

No statistics are learned from test interactions. A global chronological holdout
is used, rather than per-user splits, to simulate a real future prediction date.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.io import sha256_file, write_dataset, write_json


def _load(verified: Path, table: str) -> pd.DataFrame:
    path = Path(verified) / f'{table}.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing validated file {path}. Run validation before preparation.')
    # Explicitly prefer the audited CSV files; don't silently fall back to staged/raw data.
    return pd.read_csv(path, dtype='string', keep_default_na=False)


def prepare(verified: Path, output: Path, report_root: Path,
            train_ratio: float = 0.8, parquet: bool = True) -> dict:
    """Prepare clean entities, split interactions globally by timestamp.

    Deterministic boundary: exact floor(n * ratio) earliest interactions are
    train. If timestamp ties cross the boundary the held-out timestamp is
    moved *entirely* to test, so train.max(timestamp) < test.min(timestamp).
    This can make the actual train fraction slightly smaller than requested.
    """
    if not 0 < train_ratio < 1:
        raise ValueError('train_ratio must be strictly between 0 and 1')
    verified, output = Path(verified), Path(output)
    users = _load(verified, 'users')
    items = _load(verified, 'items')
    ratings = _load(verified, 'interactions')
    if len(ratings) < 2:
        raise ValueError('Need at least two validated interactions for a time split')
    if ratings[['user_id','item_id','rating','timestamp']].isna().any().any():
        raise ValueError('Unexpected nulls in validated interactions')
    for c in ['user_id','item_id']:
        for frame in (ratings, {'user_id': users, 'item_id': items}[c]):
            frame[c] = pd.to_numeric(frame[c], errors='raise').astype('int64')
    ratings['rating'] = pd.to_numeric(ratings['rating'], errors='raise').astype('int8')
    ratings['timestamp'] = pd.to_numeric(ratings['timestamp'], errors='raise').astype('int64')
    if not ratings['rating'].between(1, 5).all():
        raise ValueError('Ratings out of range: run validation again')
    if ratings[['user_id', 'item_id']].duplicated().any():
        raise ValueError('Duplicate user-item pair in validated input')
    if not ratings.user_id.isin(users.user_id).all() or not ratings.item_id.isin(items.item_id).all():
        raise ValueError('Orphan interaction found in validated input')
    if users.user_id.duplicated().any() or items.item_id.duplicated().any():
        raise ValueError('Duplicate entity ID in validated input')
    ratings['interaction_at_utc'] = pd.to_datetime(ratings.timestamp, unit='s', utc=True)
    ratings = ratings.sort_values(['timestamp','user_id','item_id'], kind='mergesort').reset_index(drop=True)
    boundary = int(np.floor(len(ratings) * train_ratio))
    boundary = max(1, min(boundary, len(ratings)-1))
    # Keep all events at the boundary timestamp in test; no temporal overlap.
    test_start_ts = int(ratings.iloc[boundary].timestamp)
    train = ratings.loc[ratings.timestamp < test_start_ts].copy().reset_index(drop=True)
    test = ratings.loc[ratings.timestamp >= test_start_ts].copy().reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError('Cannot form nonempty strictly chronological split; too few distinct timestamps')
    assert train.timestamp.max() < test.timestamp.min()
    # Prevent training-scoped artifacts from accidentally using the full ratings frame.
    users['age'] = pd.to_numeric(users['age'], errors='raise').astype('int16')
    # Preserve zip_code as string, including leading zeros; demographics are optional for future models.
    from src.ingestion.load import GENRES
    for genre in GENRES:
        items[genre] = pd.to_numeric(items[genre], errors='raise').astype('int8')
    items['release_date'] = pd.to_datetime(items['release_date'].replace('', pd.NA),
                                           format='%d-%b-%Y', errors='coerce')
    users = users.sort_values('user_id').reset_index(drop=True)
    items = items.sort_values('item_id').reset_index(drop=True)
    paths = {}
    for name, frame in [('users',users),('items',items),('train_interactions',train),('test_interactions',test)]:
        paths[name] = write_dataset(frame, output/name, parquet=parquet)
    train_users, train_items = set(train.user_id), set(train.item_id)
    report = {
        'stage': 'preparation', 'split_method': 'global_chronological_holdout',
        'tie_policy': 'all_boundary_timestamp_events_go_to_test',
        'requested_train_ratio': train_ratio,
        'actual_train_ratio': len(train)/len(ratings),
        'source': {name: {'path': str(verified/f'{name}.csv'), 'sha256': sha256_file(verified/f'{name}.csv')}
                   for name in ('users','items','interactions')},
        'row_counts': {'users': len(users), 'items': len(items), 'total_interactions': len(ratings),
                       'train': len(train), 'test': len(test)},
        'max_train_timestamp_utc': train.interaction_at_utc.max().isoformat(),
        'min_test_timestamp_utc': test.interaction_at_utc.min().isoformat(),
        'strict_temporal_separation': bool(train.timestamp.max() < test.timestamp.min()),
        'cold_start_test_users': int((~test.user_id.isin(train_users)).sum()),
        'cold_start_test_items': int((~test.item_id.isin(train_items)).sum()),
        'unique_cold_start_users': int(test.loc[~test.user_id.isin(train_users),'user_id'].nunique()),
        'unique_cold_start_items': int(test.loc[~test.item_id.isin(train_items),'item_id'].nunique()),
        'paths': paths,
    }
    write_json(Path(report_root)/'preparation_report.json', report)
    return report
