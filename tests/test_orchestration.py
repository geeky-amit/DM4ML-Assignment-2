"""Orchestration contract tests, including simulated Prefect futures (no server required)."""
from __future__ import annotations
import json
import sys
import types
from pathlib import Path
import pytest
from flows import recommendation_flow as orchestration
from tests.test_training import prepared,train
from tests.test_evaluation import evaluate


def test_options_and_stage_contract():
    assert len(orchestration.STAGES)==8
    assert orchestration.STAGES[0]=='download' and orchestration.STAGES[-1]=='evaluate'
    for options in [orchestration.WorkflowOptions(train_ratio=1),
                    orchestration.WorkflowOptions(factors=0),
                    orchestration.WorkflowOptions(relevance_threshold=6),
                    orchestration.WorkflowOptions(tracking_uri='./mlruns')]:
        with pytest.raises(ValueError):options.validate()
    orchestration.WorkflowOptions(no_mlflow=True).validate()
    with pytest.raises(ValueError,match='Unknown stage'):
        orchestration.execute_stage('unlisted',Path('.'),orchestration.WorkflowOptions())


def test_reproducibility_baseline_match_and_changed_inputs(source_dir,tmp_path):
    root=prepared(source_dir,tmp_path)
    train(root)
    evaluate(root)
    options=orchestration.WorkflowOptions(no_mlflow=True,factors=1,train_ratio=.67,relevance_threshold=3)
    a=orchestration.verify_reproducibility(root,options)
    b=orchestration.verify_reproducibility(root,options)
    assert a['status']=='baseline_created'
    assert b['status']=='matched_previous_run'
    assert a['output_fingerprint']==b['output_fingerprint']
    changed=orchestration.verify_reproducibility(root,orchestration.WorkflowOptions(
        no_mlflow=True,factors=2,train_ratio=.67,relevance_threshold=3))
    assert changed['status']=='new_input_baseline'


def test_reproducibility_detects_changed_model_result(source_dir,tmp_path,monkeypatch):
    root=prepared(source_dir,tmp_path);train(root);evaluate(root)
    options=orchestration.WorkflowOptions(no_mlflow=True,factors=1,train_ratio=.67,relevance_threshold=3)
    orchestration.verify_reproducibility(root,options)
    original=orchestration._artifact_fingerprint
    def altered(root,options):
        d=original(root,options)
        d['output_fingerprint']='f'*64
        return d
    monkeypatch.setattr(orchestration,'_artifact_fingerprint',altered)
    with pytest.raises(ValueError,match='Reproducibility drift'):
        orchestration.verify_reproducibility(root,options)


def test_receipts_and_failure_status(tmp_path,monkeypatch):
    options=orchestration.WorkflowOptions(no_mlflow=True)
    run_dir=tmp_path/'reports/orchestration/runs/example'
    def explode(stage,root,opts):
        if stage=='validate':raise ValueError('bad data')
        return {'ok':True}
    monkeypatch.setattr(orchestration,'execute_stage',explode)
    assert orchestration._stage_receipt('download',tmp_path,options,run_dir)['ok']
    with pytest.raises(ValueError,match='bad data'):
        orchestration._stage_receipt('validate',tmp_path,options,run_dir)
    result=orchestration._finish_report(tmp_path,run_dir,'failed',options,'start',error=ValueError('bad data'))
    assert result['completed_stages']==['download']
    assert result['stages'][-1]['stage']=='validate'
    assert result['stages'][-1]['status']=='failed'
    assert json.loads((tmp_path/'reports/orchestration/latest_run.json').read_text())['status']=='failed'


def fake_prefect(monkeypatch):
    """Simulates task submit and future dependencies without pretending Prefect itself ran."""
    events=[]
    class Future:
        def __init__(self,value=None,error=None):self.value=value;self.error=error
        def result(self):
            if self.error:raise self.error
            return self.value
    class Task:
        def __init__(self,fn,name='stage',retries=0):self.fn=fn;self.name=name;self.retries=retries
        def with_options(self,**opts):return Task(self.fn,opts['name'],opts['retries'])
        def submit(self,*args,wait_for=None):
            events.append((self.name,self.retries))
            try:
                for dependency in wait_for or []:dependency.result()
                return Future(self.fn(*args))
            except Exception as e:return Future(error=e)
    def task(*_args,**kwargs):return lambda fn: Task(fn,kwargs.get('name','stage'),kwargs.get('retries',0))
    def flow(*_args,**kwargs):return lambda fn:fn
    class Logger:
        def info(self,*_):pass
        def exception(self,*_):pass
    fake=types.SimpleNamespace(flow=flow,task=task,get_run_logger=lambda:Logger())
    monkeypatch.setitem(sys.modules,'prefect',fake)
    return events


def test_simulated_prefect_full_dependency_order(tmp_path,monkeypatch):
    events=fake_prefect(monkeypatch)
    monkeypatch.setattr(orchestration,'execute_stage',lambda stage,root,options:{'stage':stage})
    monkeypatch.setattr(orchestration,'verify_reproducibility',lambda root,options:{'status':'baseline_created'})
    report=orchestration.run_prefect_pipeline(tmp_path,orchestration.WorkflowOptions(no_mlflow=True))
    assert report['status']=='completed'
    assert report['completed_stages']==list(orchestration.STAGES)
    assert [x[0] for x in events]==[f'{i:02d}-{stage}' for i,stage in enumerate(orchestration.STAGES,1)]
    assert [x[1] for x in events]==[2]+[0]*7


def test_simulated_prefect_fails_fast_without_downstream_tasks(tmp_path,monkeypatch):
    events=fake_prefect(monkeypatch)
    executed=[]
    def runner(stage,root,opts):
        executed.append(stage)
        if stage=='validate':raise ValueError('invalid source')
        return {}
    monkeypatch.setattr(orchestration,'execute_stage',runner)
    with pytest.raises(ValueError,match='invalid source'):
        orchestration.run_prefect_pipeline(tmp_path,orchestration.WorkflowOptions(no_mlflow=True))
    # Submission can occur for blocked tasks, but *execution* stops at validation.
    assert executed==['download','ingest','validate']
    assert json.loads((tmp_path/'reports/orchestration/latest_run.json').read_text())['status']=='failed'


def test_missing_prefect_explained(monkeypatch):
    monkeypatch.setitem(sys.modules,'prefect',None)
    with pytest.raises(RuntimeError,match='Prefect 3 is required'):
        orchestration.build_prefect_flow()


def test_full_eight_stage_fixture_with_real_modules(source_dir,tmp_path,monkeypatch):
    """Actual ingestion→evaluation calls on a miniature corpus; Prefect scheduler mocked."""
    import shutil
    from src.ingestion import download as download_module
    fake_prefect(monkeypatch)
    dest=tmp_path/'data/raw/ml-100k'
    dest.mkdir(parents=True)
    for item in source_dir.iterdir():shutil.copyfile(item,dest/item.name)
    monkeypatch.setattr(download_module,'download_movielens',lambda *a,**k:{'archive_sha1':'fixture-only'})
    options=orchestration.WorkflowOptions(csv_only=True,no_mlflow=True,train_ratio=.67,
                                           factors=1,relevance_threshold=3)
    report=orchestration.run_prefect_pipeline(tmp_path,options)
    assert report['status']=='completed'
    assert report['completed_stages']==list(orchestration.STAGES)
    assert (tmp_path/'reports/evaluation/model_comparison.csv').is_file()
    assert report['reproducibility']['status']=='baseline_created'
    repeat=orchestration.run_prefect_pipeline(tmp_path,options)
    assert repeat['reproducibility']['status']=='matched_previous_run'
