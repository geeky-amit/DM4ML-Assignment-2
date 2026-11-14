from __future__ import annotations
import json
import pandas as pd
import pytest
from scipy import sparse
from src.validation.run import validate_and_save
from src.preparation.run import prepare
from src.transformation.run import transform


def _prepare(source_dir,tmp_path):
    validate_and_save(source_dir,tmp_path/'validated',tmp_path/'errors',tmp_path/'validation',parquet=False)
    prepare(tmp_path/'validated',tmp_path/'processed',tmp_path/'preparation',train_ratio=0.67,parquet=False)
    return tmp_path/'processed'


def test_aggregates_and_sparse_matrix(source_dir,tmp_path):
    prepared=_prepare(source_dir,tmp_path)
    report=transform(prepared,tmp_path/'derived',tmp_path/'transformation',parquet=False)
    users=pd.read_csv(tmp_path/'derived/user_activity.csv')
    items=pd.read_csv(tmp_path/'derived/item_popularity.csv')
    matrix=sparse.load_npz(tmp_path/'derived/user_item_ratings_train.npz')
    train=pd.read_csv(prepared/'train_interactions.csv')
    assert matrix.format=='csr'
    assert matrix.shape==(3,2)
    assert matrix.nnz==len(train)==2
    assert users.user_activity_count.sum()==len(train)
    assert items.item_popularity.sum()==len(train)
    assert report['train_interactions']==2
    assert report['aggregation_scope']=='train_only'
    assert (tmp_path/'transformation/transformation_report.json').is_file()


def test_test_interactions_cannot_leak_to_aggregates(source_dir,tmp_path):
    prepared=_prepare(source_dir,tmp_path)
    report1=transform(prepared,tmp_path/'derived',tmp_path/'report',parquet=False)
    baseline=(tmp_path/'derived/user_activity.csv').read_bytes(),(tmp_path/'derived/item_popularity.csv').read_bytes()
    # Tamper with test values. Transform must not even open this file.
    (prepared/'test_interactions.csv').write_text('invalid,contents\n999,not_a_rating\n')
    report2=transform(prepared,tmp_path/'derived',tmp_path/'report',parquet=False)
    assert baseline==((tmp_path/'derived/user_activity.csv').read_bytes(),(tmp_path/'derived/item_popularity.csv').read_bytes())
    assert report1['train_csv_sha256']==report2['train_csv_sha256']


def test_known_matrix_values_and_unseen_item(source_dir,tmp_path):
    prepared=_prepare(source_dir,tmp_path)
    transform(prepared,tmp_path/'derived',tmp_path/'report',parquet=False)
    matrix=sparse.load_npz(tmp_path/'derived/user_item_ratings_train.npz')
    assert matrix[0,0]==5
    assert matrix[0,1]==4
    assert matrix[1,0]==0   # held-out 2 -> movie 1 rating = 3 MUST NOT be present
    users=pd.read_csv(tmp_path/'derived/user_activity.csv')
    assert users.loc[users.user_id==2,'user_activity_count'].item()==0
    assert pd.isna(users.loc[users.user_id==2,'user_avg_rating']).all()


def test_missing_prepared_data(source_dir,tmp_path):
    with pytest.raises(FileNotFoundError, match='prepared'):
        transform(tmp_path/'missing',tmp_path/'derived',tmp_path/'report',parquet=False)
