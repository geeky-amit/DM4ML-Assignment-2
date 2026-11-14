"""Hand-checked metric examples, held-out integration, and provenance guards."""
import json
import subprocess
import sys
from pathlib import Path
import pytest
from src.evaluation.metrics import ranking_metrics,genre_diversity
from src.evaluation.run import evaluate_models
from tests.test_training import prepared,train


def evaluate(root, threshold=3, tracking=False):
    return evaluate_models(root/'data/processed',root/'data/features',root/'models',
                           root/'reports/training/training_report.json',root/'reports/evaluation',
                           relevant_rating=threshold,tracking=tracking)


def test_precision_recall_ndcg_hand_calculated():
    r=ranking_metrics([12,11,13],{11,13},3)
    assert r['precision']==pytest.approx(2/3)
    assert r['recall']==pytest.approx(1)
    import math
    assert r['ndcg']==pytest.approx((1/math.log2(3)+1/math.log2(4))/(1+1/math.log2(3)))
    assert ranking_metrics([11],{11,12},5)['precision']==pytest.approx(.2)
    assert ranking_metrics([99],{11},5)['ndcg']==0


def test_metric_rejects_incorrect_input():
    with pytest.raises(ValueError): ranking_metrics([1,1],{1},5)
    with pytest.raises(ValueError): ranking_metrics([1,2,3],{1},2)
    with pytest.raises(ValueError): ranking_metrics([] ,set(),5)


def test_diversity_jaccard():
    genres={1:frozenset({'action','comedy'}),2:frozenset({'action'}),3:frozenset({'drama'})}
    assert genre_diversity([1,2],genres)==pytest.approx(.5)
    assert genre_diversity([1,3],genres)==pytest.approx(1.)
    assert genre_diversity([1],genres)==0.
    assert genre_diversity([1,2,3],genres)==pytest.approx((.5+1+1)/3)
    assert genre_diversity([1,2],{1:frozenset(),2:frozenset()})==0


def test_end_to_end_comparison_and_audit(source_dir,tmp_path):
    root=prepared(source_dir,tmp_path)
    train(root)
    report=evaluate(root)
    assert report['status']=='OK'
    assert report['eligible_users']==1
    assert report['all_test_interactions']==1
    for name in ('popularity','collaborative_svd'):
        assert report['metrics'][name]['eligible_users']==1
        for k in (5,10):
            for measure in ('precision','recall','ndcg','coverage','diversity'):
                assert 0 <= report['metrics'][name][f'{measure}_at_{k}'] <= 1
    for fn in ('evaluation_report.json','model_comparison.csv','model_comparison.png','recommendation_audit.csv'):
        assert (root/'reports/evaluation'/fn).stat().st_size>0
    assert report['test_sha256']


def test_test_label_changes_evaluation_not_model(source_dir,tmp_path):
    root=prepared(source_dir,tmp_path)
    original=train(root)
    one=evaluate(root,threshold=3)
    testfile=root/'data/processed/test_interactions.csv'
    contents=testfile.read_text()
    testfile.write_text(contents.replace(',3,',',2,'))
    with pytest.raises(ValueError,match='No eligible'):
        evaluate(root,threshold=3)
    assert json.loads((root/'models/manifest.json').read_text())['artifacts']==original['artifact_sha256']


def test_model_artifact_tampering_blocks_evaluation(source_dir,tmp_path):
    root=prepared(source_dir,tmp_path)
    train(root)
    with (root/'models/collaborative_svd.npz').open('ab') as f: f.write(b'x')
    with pytest.raises(ValueError,match='checksum'):
        evaluate(root)


def test_missing_evaluation_data_rejected(source_dir,tmp_path):
    root=prepared(source_dir,tmp_path)
    train(root)
    (root/'data/processed/test_interactions.csv').unlink()
    with pytest.raises(FileNotFoundError): evaluate(root)


def test_evaluation_cli(source_dir,tmp_path):
    root=prepared(source_dir,tmp_path)
    train(root)
    repo=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'-m','scripts.pipeline','milestone5','--root',str(root),
                           '--relevance-threshold','3','--no-mlflow'],cwd=repo,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'EVALUATE OK:' in result.stdout


def test_mlflow_metric_logging_uses_existing_runs(monkeypatch):
    """Simulated MLflow client verifies logging keys to pre-existing training runs."""
    import sys,types
    from src.evaluation.run import _log_mlflow
    calls=[]
    class Client:
        def __init__(self,tracking_uri): calls.append(('uri',tracking_uri))
        def get_run(self,run_id): calls.append(('get',run_id))
        def log_metric(self,run_id,key,value): calls.append(('metric',run_id,key,value))
    mlflow=types.SimpleNamespace(set_tracking_uri=lambda value:calls.append(('set',value)))
    tracking=types.SimpleNamespace(MlflowClient=Client)
    monkeypatch.setitem(sys.modules,'mlflow',mlflow)
    monkeypatch.setitem(sys.modules,'mlflow.tracking',tracking)
    res=_log_mlflow({'popularity':{'precision_at_5':.1},'collaborative_svd':{'ndcg_at_10':.2}},
                   {'mlflow_run_ids':{'popularity':'run1','collaborative_svd':'run2'}},
                   'sqlite:///mlflow.db')
    assert res=={'popularity':'run1','collaborative_svd':'run2'}
    assert ('metric','run1','eval_precision_at_5',.1) in calls
    assert ('metric','run2','eval_ndcg_at_10',.2) in calls
