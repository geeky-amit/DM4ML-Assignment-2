"""Immutable training-as-of feature snapshots and a shared candidate feature API.

No test-interaction file is accessed. Snapshot features are valid for ranking after
training cutoff, not for predicting historical labels inside the same train data.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.ingestion.load import GENRES
from src.utils.io import sha256_file, write_dataset, write_json

FEATURE_VERSION = '1.0.0'
USER_FIELDS = ['user_id','user_activity_count','user_avg_rating','user_last_timestamp',
               'days_since_last_interaction','user_preferred_genre']
ITEM_FIELDS = ['item_id','item_popularity','item_avg_rating','item_genres']
CANDIDATE_FIELDS = ['user_id','item_id','user_activity_count','user_avg_rating',
                    'days_since_last_interaction','user_preferred_genre',
                    'item_popularity','item_avg_rating','item_genres',
                    'seen_in_training','genre_affinity']


def _read_csv(path: Path, **kw) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Missing required feature input {path}; run full/milestone2 first')
    return pd.read_csv(path, keep_default_na=False, **kw)


def _source_hashes(prepared: Path) -> dict:
    names = ['users','items','train_interactions']
    return {name: sha256_file(prepared / f'{name}.csv') for name in names}


def snapshot_id(source_hashes: dict) -> str:
    content = json.dumps({'version':FEATURE_VERSION,'sources':source_hashes},sort_keys=True)
    return 'v'+FEATURE_VERSION.replace('.','_')+'_'+hashlib.sha256(content.encode()).hexdigest()[:16]


def _build_tables(prepared: Path) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,int]:
    users = _read_csv(prepared/'users.csv')
    items = _read_csv(prepared/'items.csv')
    train = _read_csv(prepared/'train_interactions.csv')
    for frame,col in [(users,'user_id'),(items,'item_id'),(train,'user_id'),(train,'item_id')]:
        frame[col] = pd.to_numeric(frame[col],errors='raise').astype('int64')
    train['rating'] = pd.to_numeric(train.rating,errors='raise')
    train['timestamp'] = pd.to_numeric(train.timestamp,errors='raise').astype('int64')
    for frame,cols in ((users,['user_id']),(items,['item_id']),(train,['user_id','item_id'])):
        if frame.duplicated(cols).any():
            raise ValueError('Duplicate entity or train interaction key')
    if train.empty:
        raise ValueError('Cannot produce features from empty training interactions')
    if not train.user_id.isin(users.user_id).all() or not train.item_id.isin(items.item_id).all():
        raise ValueError('Orphan training interaction')
    if not train.rating.between(1,5).all():
        raise ValueError('Out of range train ratings')
    for genre in GENRES:
        items[genre] = pd.to_numeric(items[genre],errors='raise').astype('int8')
        if not items[genre].isin([0,1]).all():
            raise ValueError(f'Invalid genre indicator: {genre}')
    cutoff = int(train.timestamp.max())
    counts = train.groupby('user_id').agg(user_activity_count=('rating','size'),
                user_avg_rating=('rating','mean'),user_last_timestamp=('timestamp','max')).reset_index()
    user = users[['user_id']].merge(counts,on='user_id',how='left',validate='one_to_one')
    user['user_activity_count'] = user.user_activity_count.fillna(0).astype('int64')
    user['days_since_last_interaction'] = (cutoff-user.user_last_timestamp)/86400.0
    # Only *training-period* ratings contribute to genre preferences.
    exploded = train[['user_id','item_id','rating']].merge(items[['item_id',*GENRES]],on='item_id',validate='many_to_one')
    genre_scores = exploded.loc[:,GENRES].mul(exploded.rating,axis=0)
    genre_scores['user_id'] = exploded.user_id
    genre_totals = genre_scores.groupby('user_id')[list(GENRES)].sum()
    # Stable tie-breaking: original GENRES ordering; no-history -> empty string.
    genre_preference = genre_totals.idxmax(axis=1).rename('user_preferred_genre')
    genre_preference.loc[genre_totals.max(axis=1).eq(0)] = ''
    user = user.merge(genre_preference,on='user_id',how='left',validate='one_to_one')
    user['user_preferred_genre'] = user.user_preferred_genre.fillna('')
    user = user[USER_FIELDS].sort_values('user_id').reset_index(drop=True)
    popularity = train.groupby('item_id').agg(item_popularity=('rating','size'),
                                            item_avg_rating=('rating','mean')).reset_index()
    item = items[['item_id',*GENRES]].merge(popularity,on='item_id',how='left',validate='one_to_one')
    item['item_popularity'] = item.item_popularity.fillna(0).astype('int64')
    item['item_genres'] = item.apply(lambda row: '|'.join(g for g in GENRES if row[g] == 1),axis=1)
    item = item[ITEM_FIELDS].sort_values('item_id').reset_index(drop=True)
    seen = train[['user_id','item_id']].sort_values(['user_id','item_id']).reset_index(drop=True)
    return user,item,seen,cutoff


def _verify_snapshot(location: Path, manifest: dict) -> None:
    for name,expected in manifest['artifacts'].items():
        file = location / name
        if not file.is_file() or sha256_file(file) != expected:
            raise ValueError(f'Immutable feature snapshot corrupted: {file}')


def materialize_features(prepared: Path, feature_root: Path, report_root: Path,
                         parquet: bool = True) -> dict:
    """Build or verify immutable training-only snapshot; never read test CSV."""
    prepared, feature_root, report_root = map(Path,(prepared,feature_root,report_root))
    source = _source_hashes(prepared)
    sid = snapshot_id(source)
    location = feature_root/'snapshots'/sid
    manifest_file = location/'manifest.json'
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        if manifest.get('source_sha256') != source or manifest.get('feature_version') != FEATURE_VERSION:
            raise ValueError('Existing feature snapshot has incompatible manifest')
        _verify_snapshot(location,manifest)
    else:
        if location.exists() and any(location.iterdir()):
            raise ValueError('Partial feature snapshot exists, refusing overwrite')
        user,item,seen,cutoff = _build_tables(prepared)
        location.mkdir(parents=True,exist_ok=True)
        outputs = {}
        for name,frame in [('user_features',user),('item_features',item),('seen_train_pairs',seen)]:
            saved = write_dataset(frame,location/name,parquet=parquet)
            for fmt,path in saved.items():
                outputs[Path(path).name] = sha256_file(Path(path))
        manifest = {'feature_version':FEATURE_VERSION,'snapshot_id':sid,
                    'source_sha256':source,'as_of_timestamp':cutoff,
                    'entities':{'users':len(user),'items':len(item),'seen_pairs':len(seen)},
                    'artifacts':dict(sorted(outputs.items())),
                    'null_policy':{'averages':'NaN for cold-start users/items',
                                   'recency':'NaN for users with no training events',
                                   'preferred_genre':'empty string when no preference'}}
        write_json(manifest_file,manifest)
    write_json(feature_root/'latest.json',{'snapshot_id':sid,'relative_path':f'snapshots/{sid}'})
    report = {'stage':'feature_management','status':'OK','feature_version':FEATURE_VERSION,
              'snapshot_id':sid,'snapshot_path':str(location),'training_only':True,
              'source_sha256':source,'entities':manifest['entities'],
              'as_of_timestamp':manifest['as_of_timestamp'],
              'artifact_sha256':manifest['artifacts']}
    write_json(report_root/'feature_report.json',report)
    return report


def load_snapshot(location: Path) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,dict]:
    """One loader used by future model training and serving/prediction."""
    location = Path(location)
    manifest = json.loads((location/'manifest.json').read_text(encoding='utf-8'))
    if manifest['feature_version'] != FEATURE_VERSION:
        raise ValueError('Unsupported feature catalog version')
    _verify_snapshot(location,manifest)
    user = _read_csv(location/'user_features.csv',dtype={'user_id':'int64','user_preferred_genre':'string'})
    item = _read_csv(location/'item_features.csv',dtype={'item_id':'int64','item_genres':'string'})
    seen = _read_csv(location/'seen_train_pairs.csv',dtype={'user_id':'int64','item_id':'int64'})
    for name in ['user_avg_rating','days_since_last_interaction']:
        user[name] = pd.to_numeric(user[name],errors='coerce')
    item['item_avg_rating'] = pd.to_numeric(item['item_avg_rating'],errors='coerce')
    return user,item,seen,manifest


def candidate_features(candidates: pd.DataFrame, location: Path) -> pd.DataFrame:
    """Deterministic batch feature lookup for either training or serving.

    Input candidates MUST be selected independently of their held-out labels.
    Never use this snapshot to model its own historical training ratings: its
    aggregates already contain those labels. Use it only for post-cutoff ranking.
    """
    if not {'user_id','item_id'}.issubset(candidates.columns):
        raise ValueError('Candidate input must contain user_id and item_id')
    req = candidates[['user_id','item_id']].copy()
    for c in ['user_id','item_id']:
        req[c] = pd.to_numeric(req[c],errors='raise').astype('int64')
    if req.isna().any().any():
        raise ValueError('Null candidate key')
    user,item,seen,_ = load_snapshot(location)
    if not req.user_id.isin(user.user_id).all() or not req.item_id.isin(item.item_id).all():
        raise ValueError('Candidate references unknown user/item; use known-entity cold-start rows')
    req['_order'] = np.arange(len(req))
    joined = req.merge(user.drop(columns=['user_last_timestamp']),on='user_id',how='left',validate='many_to_one')
    joined = joined.merge(item,on='item_id',how='left',validate='many_to_one')
    joined = joined.merge(seen.assign(seen_in_training=True),on=['user_id','item_id'],how='left',validate='many_to_one')
    joined['seen_in_training'] = joined.seen_in_training.eq(True).astype(bool)
    joined['genre_affinity'] = [int(bool(pref) and pref in genres.split('|'))
                                 for pref,genres in zip(joined.user_preferred_genre,joined.item_genres)]
    return joined.sort_values('_order')[CANDIDATE_FIELDS].reset_index(drop=True)


def demonstrate_feature_consistency(location: Path, report_root: Path, parquet: bool = True) -> dict:
    """Produce *identical* batch/serving feature lookups for fixed unlabeled pairs."""
    user,item,_,_ = load_snapshot(location)
    sample = pd.MultiIndex.from_product([user.user_id.head(3),item.item_id.head(5)],
                                         names=['user_id','item_id']).to_frame(index=False)
    batch = candidate_features(sample,location)
    serving = candidate_features(sample,location)
    if not batch.equals(serving):
        raise AssertionError('Batch and prediction feature lookups differ')
    out = Path(report_root)
    write_dataset(batch,out/'candidate_feature_example',parquet=parquet)
    result = {'consistent':True,'candidate_rows':len(batch),
              'batch_and_prediction_sha256':sha256_file(out/'candidate_feature_example.csv')}
    write_json(out/'consistency_report.json',result)
    return result
