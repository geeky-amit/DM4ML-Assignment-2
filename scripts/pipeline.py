"""Milestones 1–6 CLI. Run `python -m scripts.pipeline --help`."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.ingestion.download import download_movielens
from src.ingestion.load import ingest
from src.validation.run import validate_and_save
from src.preparation.run import prepare
from src.transformation.run import transform
from src.features.run import materialize_features, demonstrate_feature_consistency
from src.training.run import train_models, load_model
from src.evaluation.run import evaluate_models


def main(argv=None):
    parser = argparse.ArgumentParser(description='DM4ML Milestones 1–6 pipeline')
    parser.add_argument('command', choices=['download', 'ingest', 'validate', 'all', 'prepare', 'transform', 'milestone2', 'full', 'features', 'milestone3', 'full3', 'train', 'milestone4', 'full4', 'recommend', 'evaluate', 'milestone5', 'full5', 'milestone6', 'full6', 'orchestrate'])
    parser.add_argument('--root', type=Path, default=Path('.'), help='Repository root')
    parser.add_argument('--archive', type=Path, help='Use pre-downloaded official ml-100k.zip')
    parser.add_argument('--csv-only', action='store_true', help='Skip Parquet (useful for test environments)')
    parser.add_argument('--now', default='2026-11-12T00:00:00Z', help='UTC validation reference time')
    parser.add_argument('--require-clean', action='store_true', help='Exit nonzero if any data is quarantined')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Chronological training fraction, default 0.8')
    parser.add_argument('--factors', type=int, default=32, help='SVD latent factors')
    parser.add_argument('--seed', type=int, default=42, help='SVD deterministic seed')
    parser.add_argument('--no-mlflow', action='store_true', help='Offline training; MLflow runs not logged')
    parser.add_argument('--tracking-uri', default=None, help='Override MLflow tracking URI')
    parser.add_argument('--user-id', type=int, default=1, help='Recommendation example user')
    parser.add_argument('--k', type=int, default=5, help='Recommendation list length')
    parser.add_argument('--relevance-threshold', type=int, default=4, help='Evaluation relevance rating threshold (1..5)')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    raw = root / 'data/raw/ml-100k'
    try:
        if args.command in ('download', 'all', 'full', 'full3', 'full4', 'full5'):
            result = download_movielens(root / 'data/raw', args.archive)
            print('DOWNLOAD OK: official archive SHA1 verified')
            print(json.dumps(result, indent=2))
        if args.command in ('ingest', 'all', 'full', 'full3', 'full4', 'full5'):
            result = ingest(raw, root / 'data/staged', parquet=not args.csv_only)
            print('INGEST OK:', json.dumps({k:v['rows'] for k,v in result['tables'].items()}))
        if args.command in ('validate', 'all', 'full', 'full3', 'full4', 'full5'):
            result = validate_and_save(raw, root / 'data/validated', root / 'data/errors',
                                       root / 'reports/validation', parquet=not args.csv_only,
                                       now=args.now)
            print('VALIDATION:', result['status'])
            print(json.dumps(result['tables'], indent=2))
            if args.require_clean and result['total_rejected_rows']:
                raise SystemExit(2)
        if args.command in ('prepare', 'milestone2', 'full', 'full3', 'full4', 'full5'):
            prep = prepare(root / 'data/validated', root / 'data/processed',
                           root / 'reports/preparation', train_ratio=args.train_ratio,
                           parquet=not args.csv_only)
            print('PREPARE OK:', json.dumps(prep['row_counts']))
            print('TEMPORAL SEPARATION:', prep['strict_temporal_separation'])
        if args.command in ('transform', 'milestone2', 'full', 'full3', 'full4', 'full5'):
            transformed = transform(root / 'data/processed', root / 'data/processed/derived',
                                    root / 'reports/transformation', parquet=not args.csv_only)
            print('TRANSFORM OK:', json.dumps(transformed['matrix']))
        if args.command in ('features', 'milestone3', 'full3', 'full4', 'full5'):
            features = materialize_features(root / 'data/processed',root / 'data/features',
                                            root / 'reports/features',parquet=not args.csv_only)
            print('FEATURES OK:', json.dumps({'snapshot_id':features['snapshot_id'],
                  'entities':features['entities'],'training_only':features['training_only']}))
            demo = demonstrate_feature_consistency(Path(features['snapshot_path']),
                                                  root / 'reports/features', parquet=not args.csv_only)
            print('FEATURE CONSISTENCY:', demo['consistent'])
        if args.command in ('train', 'milestone4', 'full4', 'full5'):
            training = train_models(root/'data/processed',root/'data/processed/derived',
                                    root/'data/features',root/'models',root/'reports/training',
                                    n_factors=args.factors,seed=args.seed,
                                    tracking=not args.no_mlflow,tracking_uri=args.tracking_uri)
            print('TRAIN OK:',json.dumps({k:training[k] for k in
                  ('models','snapshot_id','train_interactions','actual_factors','mlflow_run_ids')}))
        if args.command in ('evaluate', 'milestone5', 'full5'):
            evaluation = evaluate_models(root/'data/processed',root/'data/features',root/'models',
                                         root/'reports/training/training_report.json',
                                         root/'reports/evaluation',relevant_rating=args.relevance_threshold,
                                         tracking=not args.no_mlflow,tracking_uri=args.tracking_uri)
            print('EVALUATE OK:',json.dumps({'eligible_users':evaluation['eligible_users'],
                'test_interactions':evaluation['all_test_interactions'],
                'metrics':evaluation['metrics'],'mlflow_run_ids':evaluation['mlflow_run_ids']}))
        if args.command in ('milestone6','full6','orchestrate'):
            from flows.recommendation_flow import WorkflowOptions, run_prefect_pipeline
            options=WorkflowOptions(archive=str(args.archive) if args.archive else None,
                csv_only=args.csv_only, now=args.now, require_clean=True,
                train_ratio=args.train_ratio, factors=args.factors, seed=args.seed,
                no_mlflow=args.no_mlflow,
                tracking_uri=args.tracking_uri or 'sqlite:///mlflow.db',
                relevance_threshold=args.relevance_threshold)
            report=run_prefect_pipeline(root,options)
            print('ORCHESTRATION OK:',json.dumps({'status':report['status'],
                'completed_stages':report['completed_stages'],
                'reproducibility':report['reproducibility']['status'],
                'run_id':report['run_id']}))
        if args.command == 'recommend':
            for model_name in ('popularity','collaborative_svd'):
                model = load_model(root/'models',model_name)
                print('RECOMMEND',model_name,':',json.dumps(model.recommend(args.user_id,args.k)))
    except Exception as exc:
        parser.exit(1, f'ERROR ({type(exc).__name__}): {exc}\n')


if __name__ == '__main__':
    main()
