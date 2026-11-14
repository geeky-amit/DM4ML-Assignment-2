"""Feature snapshot regression and invariance tests."""
import json
import pandas as pd
import pytest
from src.validation.run import validate_and_save
from src.preparation.run import prepare
from src.features.run import (materialize_features,load_snapshot,candidate_features,
                              demonstrate_feature_consistency)


def make_data(source_dir,tmp_path):
    validate_and_save(source_dir,tmp_path/'validated',tmp_path/'errors',tmp_path/'validation',parquet=False)
    prepare(tmp_path/'validated',tmp_path/'processed',tmp_path/'preparation',train_ratio=.67,parquet=False)
    return tmp_path/'processed'


def create(source_dir,tmp_path):
    p=make_data(source_dir,tmp_path)
    r=materialize_features(p,tmp_path/'features',tmp_path/'report',parquet=False)
    return p,r


def test_snapshot_entities_and_training_only(source_dir,tmp_path):
    p,r=create(source_dir,tmp_path)
    user,item,seen,meta=load_snapshot(r['snapshot_path'])
    train=pd.read_csv(p/'train_interactions.csv')
    assert r['training_only'] is True
    assert len(user)==3 and len(item)==2 and len(seen)==len(train)==2
    assert user.user_activity_count.sum()==len(train)
    assert item.item_popularity.sum()==len(train)
    assert user.loc[user.user_id==2,'user_activity_count'].item()==0
    assert pd.isna(user.loc[user.user_id==2,'user_avg_rating']).all()
    assert meta['source_sha256']['train_interactions']


def test_idempotent_immutable_snapshot(source_dir,tmp_path):
    p,r=create(source_dir,tmp_path)
    r2=materialize_features(p,tmp_path/'features',tmp_path/'report',parquet=False)
    assert r2['snapshot_id']==r['snapshot_id']
    with open(f"{r['snapshot_path']}/user_features.csv",'a') as f: f.write('tampered\n')
    with pytest.raises(ValueError,match='corrupted'):
        materialize_features(p,tmp_path/'features',tmp_path/'report',parquet=False)


def test_test_data_cannot_change_feature_snapshot(source_dir,tmp_path):
    p,r=create(source_dir,tmp_path)
    (p/'test_interactions.csv').write_text('broken,test,values\n')
    r2=materialize_features(p,tmp_path/'features',tmp_path/'report',parquet=False)
    assert r2['snapshot_id']==r['snapshot_id']


def test_candidate_features_and_consistency(source_dir,tmp_path):
    p,r=create(source_dir,tmp_path)
    loc=r['snapshot_path']
    candidates=pd.DataFrame({'user_id':[1,2,1], 'item_id':[1,1,2]})
    actual=candidate_features(candidates,loc)
    assert list(actual[['user_id','item_id']].itertuples(index=False,name=None))==[(1,1),(2,1),(1,2)]
    assert actual.seen_in_training.tolist()==[True,False,True]
    assert actual.user_activity_count.tolist()==[2,0,2]
    result=demonstrate_feature_consistency(loc,tmp_path/'report',parquet=False)
    assert result['consistent'] and result['candidate_rows']==6


def test_reject_invalid_candidates(source_dir,tmp_path):
    _,r=create(source_dir,tmp_path)
    loc=r['snapshot_path']
    with pytest.raises(ValueError,match='unknown'):
        candidate_features(pd.DataFrame({'user_id':[999], 'item_id':[1]}),loc)
    with pytest.raises(ValueError,match='must contain'):
        candidate_features(pd.DataFrame({'user_id':[1]}),loc)


def test_train_change_creates_new_version(source_dir,tmp_path):
    p,r=create(source_dir,tmp_path)
    train=pd.read_csv(p/'train_interactions.csv')
    train.loc[0,'rating']=4 if train.loc[0,'rating']!=4 else 3
    train.to_csv(p/'train_interactions.csv',index=False)
    r2=materialize_features(p,tmp_path/'features',tmp_path/'report',parquet=False)
    assert r['snapshot_id']!=r2['snapshot_id']
    assert (tmp_path/'features/snapshots'/r['snapshot_id']).exists()


def test_no_genre_pref_leakage(source_dir,tmp_path):
    p,r=create(source_dir,tmp_path)
    user,_,_,_=load_snapshot(r['snapshot_path'])
    assert user.loc[user.user_id==2,'user_preferred_genre'].item()==''


def test_milestone3_cli_from_prepared_fixture(source_dir,tmp_path):
    import subprocess,sys
    from pathlib import Path
    p=make_data(source_dir,tmp_path)
    import shutil
    shutil.copytree(p,tmp_path/'data/processed')
    repo=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'-m','scripts.pipeline','milestone3',
                           '--root',str(tmp_path),'--csv-only'],cwd=repo,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'FEATURES OK' in result.stdout
    assert 'FEATURE CONSISTENCY: True' in result.stdout
    assert (tmp_path/'data/features/latest.json').exists()
    assert (tmp_path/'reports/features/consistency_report.json').exists()
