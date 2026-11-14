"""Offline training and experiment tracking integration tests (no external server)."""
import json
import sys
import types
from contextlib import contextmanager
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from src.validation.run import validate_and_save
from src.preparation.run import prepare
from src.transformation.run import transform
from src.features.run import materialize_features
from src.training.recommenders import fit_models
from src.training.run import train_models,load_model


def prepared(source_dir,tmp_path):
    root=tmp_path
    validate_and_save(source_dir,root/'data/validated',root/'data/errors',root/'reports/validation',parquet=False)
    prepare(root/'data/validated',root/'data/processed',root/'reports/preparation',train_ratio=.67,parquet=False)
    transform(root/'data/processed',root/'data/processed/derived',root/'reports/transformation',parquet=False)
    materialize_features(root/'data/processed',root/'data/features',root/'reports/features',parquet=False)
    return root


def train(root,tracking=False):
    return train_models(root/'data/processed',root/'data/processed/derived',root/'data/features',
                        root/'models',root/'reports/training',n_factors=1,seed=42,tracking=tracking)


def test_both_models_and_persistent_loading(source_dir,tmp_path):
    root=prepared(source_dir,tmp_path)
    report=train(root)
    assert report['models']==['popularity','collaborative_svd']
    assert report['train_interactions']==2
    for kind in report['models']:
        model=load_model(root/'models',kind)
        rec=model.recommend(1,5)
        assert rec==[]  # fixture: first user has already seen all two movies
        cold=model.recommend(2,10)
        assert len(cold)==2
        assert all(r['fallback'] for r in cold)
        assert [r['item_id'] for r in cold]==[1,2]  # tie broken by ID
        assert model.recommend(999,10)==cold
    assert load_model(root/'models','collaborative_svd').kind=='collaborative_svd'


def test_known_personalization_and_seen_filter():
    m=sparse.csr_matrix(np.array([[5,0,0],[0,4,0],[0,0,5],[4,0,0]],dtype='float32'))
    baseline,cf,_=fit_models(m,np.array([1,2,3,4]),np.array([10,11,12]),n_factors=2,seed=123)
    assert 10 not in [r['item_id'] for r in cf.recommend(1,10)]
    assert all(not r['fallback'] for r in cf.recommend(1))
    assert all(r['fallback'] for r in cf.recommend(999))
    assert baseline.recommend(1)==baseline.recommend(4)
    with pytest.raises(ValueError,match='positive'):
        cf.recommend(1,k=0)


def test_reproducible_scores_and_tie_order():
    m=sparse.csr_matrix(np.array([[5,0,1],[0,4,3],[1,0,5]],dtype='float32'))
    _,a,_=fit_models(m,np.array([1,2,3]),np.array([1,2,3]),2,42)
    _,b,_=fit_models(m,np.array([1,2,3]),np.array([1,2,3]),2,42)
    np.testing.assert_allclose(a.user_factors @ a.item_factors.T,b.user_factors @ b.item_factors.T,rtol=1e-9,atol=1e-9)
    assert a.recommend(1)==b.recommend(1)


def test_artifact_tampering_rejected(source_dir,tmp_path):
    root=prepared(source_dir,tmp_path)
    train(root)
    with open(root/'models/popularity.npz','ab') as out:
        out.write(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        load_model(root/'models','popularity')


def test_train_changes_after_snapshot_rejected(source_dir,tmp_path):
    root=prepared(source_dir,tmp_path)
    path=root/'data/processed/train_interactions.csv'
    path.write_text(path.read_text()+'1,2,5,876543210\n')
    with pytest.raises(ValueError,match='changed since feature snapshot'):
        train(root)


def test_heldout_labels_cannot_affect_model(source_dir,tmp_path):
    root=prepared(source_dir,tmp_path)
    a=train(root)
    original=load_model(root/'models','collaborative_svd').recommend(999,10)
    # Invalid test contents cannot influence training: training does not open this file.
    (root/'data/processed/test_interactions.csv').write_text('not,valid,test\n')
    b=train(root)
    assert a['artifact_sha256']==b['artifact_sha256']
    assert load_model(root/'models','collaborative_svd').recommend(999,10)==original


def test_tracking_calls_and_run_ids(source_dir,tmp_path,monkeypatch):
    root=prepared(source_dir,tmp_path)
    calls=[]
    @contextmanager
    def start_run(run_name):
        calls.append(('start',run_name))
        yield types.SimpleNamespace(info=types.SimpleNamespace(run_id=f'run{len([c for c in calls if c[0]=="start"])}'))
    fake=types.SimpleNamespace(set_tracking_uri=lambda uri:calls.append(('uri',uri)),
                               set_experiment=lambda name:calls.append(('experiment',name)),
                               start_run=start_run,
                               log_params=lambda p:calls.append(('params',p)),
                               log_metrics=lambda m:calls.append(('metrics',m)),
                               log_artifact=lambda p,artifact_path=None:calls.append(('artifact',p,artifact_path)))
    monkeypatch.setitem(sys.modules,'mlflow',fake)
    r=train(root,tracking=True)
    assert set(r['mlflow_run_ids'])=={'popularity','collaborative_svd'}
    assert len([c for c in calls if c[0]=='start'])==2
    assert len([c for c in calls if c[0]=='artifact'])==4
    assert all(c[1]['feature_snapshot']==r['snapshot_id'] for c in calls if c[0]=='params')


def test_invalid_factor_and_matrix():
    with pytest.raises(ValueError,match='positive integer'):
        fit_models(sparse.csr_matrix(np.eye(2)),np.array([1,2]),np.array([1,2]),n_factors=0)
    with pytest.raises(ValueError,match='dimensions'):
        fit_models(sparse.csr_matrix(np.eye(2)),np.array([1]),np.array([1,2]))
