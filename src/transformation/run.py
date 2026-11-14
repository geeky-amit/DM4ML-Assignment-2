"""Training-only aggregates and sparse user–item rating matrix.

Never load test interactions when deriving features/aggregates. In particular,
items/users with no training history have zero activity, rather than a future-
aware value computed from the full interaction history.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from src.utils.io import sha256_file, write_dataset, write_json


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Missing prepared data {path}. Run prepare first.')
    return pd.read_csv(path, dtype={'user_id':'int64','item_id':'int64'}, keep_default_na=False)


def transform(prepared: Path, output: Path, report_root: Path,
              parquet: bool = True) -> dict:
    """Create training-only user/movie summaries and sparse CSR rating matrix."""
    prepared, output = Path(prepared), Path(output)
    train_path = prepared/'train_interactions.csv'
    users, items, train = (_read(prepared/f'{n}.csv') for n in ('users','items','train_interactions'))
    if train.empty:
        raise ValueError('Empty train split')
    if train[['user_id','item_id']].duplicated().any():
        raise ValueError('Duplicate user-item pairs cannot form a valid sparse matrix')
    if not train.user_id.isin(users.user_id).all() or not train.item_id.isin(items.item_id).all():
        raise ValueError('Train interactions contain unknown entities')
    train['timestamp'] = pd.to_numeric(train.timestamp, errors='raise').astype('int64')
    train['rating'] = pd.to_numeric(train.rating, errors='raise').astype('float64')
    cutoff = pd.to_datetime(train.timestamp.max(), unit='s', utc=True)
    user_counts = train.groupby('user_id').agg(user_activity_count=('rating','size'),
                                               user_avg_rating=('rating','mean'),
                                               user_last_timestamp=('timestamp','max')).reset_index()
    user_summary = users[['user_id']].merge(user_counts, on='user_id', how='left', validate='one_to_one')
    user_summary['user_activity_count'] = user_summary.user_activity_count.fillna(0).astype('int64')
    user_summary['days_since_last_interaction'] = (train.timestamp.max() - user_summary.user_last_timestamp) / 86400.0
    item_counts = train.groupby('item_id').agg(item_popularity=('rating','size'),
                                               item_avg_rating=('rating','mean')).reset_index()
    item_summary = items[['item_id','title']].merge(item_counts, on='item_id', how='left', validate='one_to_one')
    item_summary['item_popularity'] = item_summary.item_popularity.fillna(0).astype('int64')
    # Missing means "no training observation"; don't replace with a future-aware global mean.
    user_mapping = users[['user_id']].sort_values('user_id').reset_index(drop=True)
    user_mapping['row_index'] = np.arange(len(user_mapping), dtype='int64')
    item_mapping = items[['item_id']].sort_values('item_id').reset_index(drop=True)
    item_mapping['column_index'] = np.arange(len(item_mapping), dtype='int64')
    indexed = train[['user_id','item_id','rating']].merge(user_mapping, on='user_id', validate='many_to_one').merge(
        item_mapping, on='item_id', validate='many_to_one')
    matrix = sparse.coo_matrix((indexed.rating.to_numpy(dtype='float32'),
                               (indexed.row_index.to_numpy(), indexed.column_index.to_numpy())),
                              shape=(len(user_mapping),len(item_mapping)), dtype='float32').tocsr()
    if matrix.nnz != len(train):
        raise AssertionError('Matrix entries must equal training interactions')
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in [('user_activity',user_summary),('item_popularity',item_summary),
                        ('user_index',user_mapping),('item_index',item_mapping)]:
        paths[name] = write_dataset(frame, output/name, parquet=parquet)
    matrix_path = output/'user_item_ratings_train.npz'
    sparse.save_npz(matrix_path, matrix)
    report = {'stage':'transformation','aggregation_scope':'train_only',
              'train_csv_sha256':sha256_file(train_path),
              'feature_as_of_utc':cutoff.isoformat(),
              'users':len(users),'items':len(items),'train_interactions':len(train),
              'users_with_train_history':int((user_summary.user_activity_count>0).sum()),
              'items_with_train_history':int((item_summary.item_popularity>0).sum()),
              'matrix':{'format':'csr','shape':[int(v) for v in matrix.shape],
                        'nonzero_entries':int(matrix.nnz), 'dtype':str(matrix.dtype),
                        'path':str(matrix_path), 'sha256':sha256_file(matrix_path)},
              'paths':paths}
    write_json(Path(report_root)/'transformation_report.json',report)
    return report
