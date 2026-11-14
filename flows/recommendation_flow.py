"""Prefect 3 orchestration for DM4ML. Lazy Prefect import preserves legacy CLIs.

Run from repository root with: python -m scripts.pipeline milestone6
The Prefect server/UI is optional for execution, but required for UI evidence.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.io import sha256_file, write_json

STAGES = (
    'download', 'ingest', 'validate', 'prepare', 'transform',
    'features', 'train', 'evaluate',
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class WorkflowOptions:
    archive: str | None = None
    csv_only: bool = False
    now: str = '2026-11-12T00:00:00Z'
    require_clean: bool = True
    train_ratio: float = 0.8
    factors: int = 32
    seed: int = 42
    no_mlflow: bool = False
    tracking_uri: str = 'sqlite:///mlflow.db'
    relevance_threshold: int = 4

    def validate(self):
        if not 0 < self.train_ratio < 1:
            raise ValueError('train_ratio must be between 0 and 1')
        if self.factors < 1:
            raise ValueError('factors must be positive')
        if not 1 <= self.relevance_threshold <= 5:
            raise ValueError('relevance_threshold must be in 1..5')
        if not self.no_mlflow and not self.tracking_uri.startswith('sqlite:///'):
            raise ValueError('Milestone 6 requires a local SQLite MLflow URI')


def execute_stage(stage: str, root: Path, options: WorkflowOptions) -> dict[str, Any]:
    """Pure stage dispatcher shared by real Prefect tasks and fixture-based tests."""
    if stage not in STAGES:
        raise ValueError(f'Unknown stage: {stage}')
    root = Path(root).resolve()
    raw = root / 'data/raw/ml-100k'
    parquet = not options.csv_only
    if stage == 'download':
        from src.ingestion.download import download_movielens
        output = download_movielens(root / 'data/raw', Path(options.archive) if options.archive else None)
        return {'archive_sha1': output['archive_sha1']}
    if stage == 'ingest':
        from src.ingestion.load import ingest
        output = ingest(raw, root / 'data/staged', parquet=parquet)
        return {'rows': {name: value['rows'] for name, value in output['tables'].items()}}
    if stage == 'validate':
        from src.validation.run import validate_and_save
        output = validate_and_save(raw, root / 'data/validated', root / 'data/errors',
                                   root / 'reports/validation', parquet=parquet, now=options.now)
        if options.require_clean and output['total_rejected_rows']:
            raise ValueError(f"Validation quarantined {output['total_rejected_rows']} rows")
        return {'status': output['status'], 'rejected_rows': output['total_rejected_rows']}
    if stage == 'prepare':
        from src.preparation.run import prepare
        output = prepare(root / 'data/validated', root / 'data/processed',
                         root / 'reports/preparation', train_ratio=options.train_ratio, parquet=parquet)
        if not output['strict_temporal_separation']:
            raise ValueError('Train/test time boundary is not strictly separated')
        return {'rows': output['row_counts'], 'temporal_separation': True}
    if stage == 'transform':
        from src.transformation.run import transform
        output = transform(root / 'data/processed', root / 'data/processed/derived',
                           root / 'reports/transformation', parquet=parquet)
        return {'matrix': {'shape': output['matrix']['shape'], 'nonzero_entries': output['matrix']['nonzero_entries']}}
    if stage == 'features':
        from src.features.run import materialize_features, demonstrate_feature_consistency
        output = materialize_features(root / 'data/processed', root / 'data/features',
                                      root / 'reports/features', parquet=parquet)
        consistent = demonstrate_feature_consistency(Path(output['snapshot_path']),
                     root / 'reports/features', parquet=parquet)
        if not consistent['consistent'] or not output['training_only']:
            raise ValueError('Feature train/prediction consistency or leakage guard failed')
        return {'snapshot_id': output['snapshot_id'], 'training_only': True, 'consistent': True}
    if stage == 'train':
        from src.training.run import train_models
        output = train_models(root/'data/processed', root/'data/processed/derived',
                              root/'data/features',root/'models',root/'reports/training',
                              n_factors=options.factors, seed=options.seed,
                              tracking=not options.no_mlflow, tracking_uri=options.tracking_uri)
        return {'models': output['models'], 'run_ids': output['mlflow_run_ids'],
                'snapshot_id': output['snapshot_id'], 'train_interactions': output['train_interactions']}
    from src.evaluation.run import evaluate_models
    output = evaluate_models(root/'data/processed',root/'data/features',root/'models',
                             root/'reports/training/training_report.json',root/'reports/evaluation',
                             relevant_rating=options.relevance_threshold,
                             tracking=not options.no_mlflow,tracking_uri=options.tracking_uri)
    return {'eligible_users':output['eligible_users'], 'models':list(output['metrics']),
            'test_interactions': output['all_test_interactions'], 'run_ids':output['mlflow_run_ids']}


def _artifact_fingerprint(root: Path, options: WorkflowOptions) -> dict[str, str]:
    """Reproduce decisions from artifact *contents*, ignoring wall clock and MLflow run IDs."""
    from src.training.run import load_model
    root = Path(root)
    train = root/'data/processed/train_interactions.csv'
    test = root/'data/processed/test_interactions.csv'
    latest = json.loads((root/'data/features/latest.json').read_text())
    evaluation = json.loads((root/'reports/evaluation/evaluation_report.json').read_text())
    inputs = {'train_sha256':sha256_file(train),'test_sha256':sha256_file(test),
              'feature_snapshot':latest['snapshot_id'],
              'factors':options.factors,'seed':options.seed,
              'train_ratio':options.train_ratio,'relevance_threshold':options.relevance_threshold}
    model_array_hashes = {}
    for kind in ('popularity','collaborative_svd'):
        model = load_model(root/'models',kind)  # verifies integrity hashes
        h = hashlib.sha256()
        for value in (model.user_ids, model.item_ids, model.popularity,
                      model.seen.data, model.seen.indices, model.seen.indptr,
                      model.user_factors,model.item_factors):
            if value is not None:
                import numpy as np
                a = np.ascontiguousarray(value)
                h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        model_array_hashes[kind] = h.hexdigest()
    outputs={'model_arrays':model_array_hashes,'metrics':evaluation['metrics'],
             'eligible_users':evaluation['eligible_users']}
    canonical = lambda obj: json.dumps(obj,sort_keys=True,separators=(',',':')).encode()
    return {'input_fingerprint':hashlib.sha256(canonical(inputs)).hexdigest(),
            'output_fingerprint':hashlib.sha256(canonical(outputs)).hexdigest(),
            'input_details':inputs,'model_array_sha256':model_array_hashes}


def verify_reproducibility(root: Path, options: WorkflowOptions) -> dict:
    """Compare stable numeric results to prior successful execution with identical inputs."""
    root = Path(root)
    folder = root/'reports/orchestration'
    folder.mkdir(parents=True,exist_ok=True)
    latest = _artifact_fingerprint(root,options)
    baseline_file=folder/'reproducibility_baseline.json'
    status='baseline_created'
    if baseline_file.exists():
        baseline=json.loads(baseline_file.read_text())
        if baseline['input_fingerprint']==latest['input_fingerprint']:
            if baseline['output_fingerprint']!=latest['output_fingerprint']:
                raise ValueError('Reproducibility drift: same inputs/configuration, different model/evaluation outputs')
            status='matched_previous_run'
        else:
            status='new_input_baseline'
    write_json(baseline_file,latest)
    return {'status':status,**latest}


def _stage_receipt(stage: str, root: Path, options: WorkflowOptions, run_dir: Path) -> dict:
    path=Path(run_dir)/f'{STAGES.index(stage)+1:02d}_{stage}.json'
    begin=_utc_now()
    try:
        result=execute_stage(stage,Path(root),options)
        receipt={'stage':stage,'status':'completed','started_utc':begin,'ended_utc':_utc_now(),
                 'result':result}
        write_json(path,receipt)
        return result
    except Exception as exc:
        write_json(path,{'stage':stage,'status':'failed','started_utc':begin,
                         'ended_utc':_utc_now(),'error_type':type(exc).__name__,'error':str(exc)})
        raise


def _finish_report(root: Path, run_dir: Path, status: str, options: WorkflowOptions,
                   started: str, error: Exception | None = None, reproducibility: dict | None = None) -> dict:
    receipts=[]
    for stage in STAGES:
        file=run_dir/f'{STAGES.index(stage)+1:02d}_{stage}.json'
        if file.exists():
            receipts.append(json.loads(file.read_text()))
    report={'flow':'dm4ml-recommendation-end-to-end', 'run_id':run_dir.name,
            'status':status,'started_utc':started,'ended_utc':_utc_now(),
            'options':asdict(options),'stage_order':list(STAGES),'stages':receipts,
            'completed_stages':[r['stage'] for r in receipts if r['status']=='completed']}
    if reproducibility is not None:
        report['reproducibility']=reproducibility
    if error is not None:
        report['error_type']=type(error).__name__
        report['error']=str(error)
    write_json(run_dir/'run_report.json',report)
    write_json(Path(root)/'reports/orchestration/latest_run.json',report)
    return report


def build_prefect_flow():
    """Import Prefect lazily so older CLIs and unit tests remain usable without it."""
    try:
        from prefect import flow,task,get_run_logger
    except ImportError as exc:
        raise RuntimeError('Prefect 3 is required for milestone6. Install: python -m pip install -r requirements.txt') from exc

    @task(name='DM4ML-stage',retries=0,log_prints=True)
    def stage_task(stage: str, root: str, options: WorkflowOptions, run_dir: str) -> dict:
        logger=get_run_logger()
        logger.info('Starting %s',stage)
        try:
            result=_stage_receipt(stage,Path(root),options,Path(run_dir))
        except Exception:
            logger.exception('Stage failed: %s',stage)
            raise
        logger.info('Completed %s: %s',stage,result)
        return result

    @flow(name='dm4ml-recommendation-end-to-end',log_prints=True)
    def pipeline_flow(root: str, options: WorkflowOptions) -> dict:
        options.validate()
        root_path=Path(root).resolve()
        run_dir=root_path/'reports/orchestration/runs'/uuid.uuid4().hex
        run_dir.mkdir(parents=True,exist_ok=False)
        started=_utc_now()
        last=None
        try:
            for order,stage in enumerate(STAGES,1):
                # Only acquisition is retried; deterministic validation/data failures fail fast.
                configured=stage_task.with_options(name=f'{order:02d}-{stage}',
                                                    retries=2 if stage=='download' else 0,
                                                    retry_delay_seconds=5)
                last=configured.submit(stage,str(root_path),options,str(run_dir),
                                       wait_for=[last] if last is not None else [])
            last.result()  # propagate the failing stage's exception to Prefect flow state
            repro=verify_reproducibility(root_path,options)
            return _finish_report(root_path,run_dir,'completed',options,started,reproducibility=repro)
        except Exception as exc:
            _finish_report(root_path,run_dir,'failed',options,started,error=exc)
            raise

    return pipeline_flow


def run_prefect_pipeline(root: Path, options: WorkflowOptions) -> dict:
    options.validate()
    return build_prefect_flow()(str(Path(root).resolve()),options)


if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser(description='Run all 8 DM4ML stages under Prefect 3')
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--archive',type=Path)
    ap.add_argument('--no-mlflow',action='store_true')
    ap.add_argument('--tracking-uri',default='sqlite:///mlflow.db')
    ns=ap.parse_args()
    result=run_prefect_pipeline(ns.root,WorkflowOptions(archive=str(ns.archive) if ns.archive else None,
                               no_mlflow=ns.no_mlflow,tracking_uri=ns.tracking_uri))
    print('ORCHESTRATION OK:',json.dumps({'status':result['status'],
          'completed_stages':result['completed_stages'],
          'reproducibility':result.get('reproducibility',{}).get('status')}))
