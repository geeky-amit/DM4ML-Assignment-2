from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import pytest
from src.validation.run import validate_and_save
from src.preparation.run import prepare


def _fixture(source_dir, tmp_path, rows=None):
    if rows is not None:
        (source_dir/'u.data').write_text('\n'.join('\t'.join(map(str,r)) for r in rows)+'\n')
    validate_and_save(source_dir,tmp_path/'validated',tmp_path/'errors',tmp_path/'validation',parquet=False)
    return tmp_path/'validated'


def test_chronology_and_counts(source_dir,tmp_path):
    verified=_fixture(source_dir,tmp_path)
    report=prepare(verified,tmp_path/'processed',tmp_path/'report',train_ratio=0.67,parquet=False)
    train=pd.read_csv(tmp_path/'processed/train_interactions.csv')
    test=pd.read_csv(tmp_path/'processed/test_interactions.csv')
    assert len(train)+len(test)==3
    assert train.timestamp.max()<test.timestamp.min()
    assert train['interaction_at_utc'].str.endswith('+00:00').all()
    assert report['strict_temporal_separation']
    assert report['row_counts']['total_interactions']==3
    assert (tmp_path/'processed/users.csv').exists()
    assert (tmp_path/'processed/items.csv').exists()


def test_tie_split_has_no_cross_boundary_timestamp(source_dir,tmp_path):
    rows=[(1,1,5,875000000),(1,2,4,875000001),(2,1,3,875000001)]
    verified=_fixture(source_dir,tmp_path,rows)
    report=prepare(verified,tmp_path/'processed',tmp_path/'report',train_ratio=0.67,parquet=False)
    assert report['row_counts']['train']==1
    assert report['row_counts']['test']==2


def test_impossible_split_rejected(source_dir,tmp_path):
    rows=[(1,1,5,875000000),(1,2,4,875000000),(2,1,3,875000000)]
    verified=_fixture(source_dir,tmp_path,rows)
    with pytest.raises(ValueError, match='strictly chronological'):
        prepare(verified,tmp_path/'processed',tmp_path/'report',parquet=False)


def test_invalid_ratio_and_missing_validated(source_dir,tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare(tmp_path/'missing',tmp_path/'out',tmp_path/'report',parquet=False)
    verified=_fixture(source_dir,tmp_path)
    for val in (0,1,-0.1,1.1):
        with pytest.raises(ValueError, match='train_ratio'):
            prepare(verified,tmp_path/'processed',tmp_path/'report',train_ratio=val,parquet=False)


def test_cold_start_diagnostic(source_dir,tmp_path):
    rows=[(1,1,5,875000000),(1,2,4,875000001),(2,1,3,875000002)]
    verified=_fixture(source_dir,tmp_path,rows)
    report=prepare(verified,tmp_path/'processed',tmp_path/'report',train_ratio=0.67,parquet=False)
    assert report['cold_start_test_users']==1
    assert report['unique_cold_start_users']==1


def test_rejects_tampered_validated_file(source_dir,tmp_path):
    verified=_fixture(source_dir,tmp_path)
    df=pd.read_csv(verified/'interactions.csv')
    df.loc[0,'item_id']=999
    df.to_csv(verified/'interactions.csv',index=False)
    with pytest.raises(ValueError, match='Orphan'):
        prepare(verified,tmp_path/'processed',tmp_path/'report',parquet=False)
