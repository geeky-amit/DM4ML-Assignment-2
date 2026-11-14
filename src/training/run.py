"""Model training, integrity-checked artifacts and optional local MLflow experiments."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from src.features.run import load_snapshot
from src.training.recommenders import Recommender,fit_models
from src.utils.io import sha256_file,write_json

MODEL_VERSION='1.0.0'


def _snapshot_path(feature_root: Path) -> Path:
    latest=json.loads((feature_root/'latest.json').read_text(encoding='utf-8'))
    root=feature_root.resolve()
    location=(root/latest['relative_path']).resolve()
    if not location.is_relative_to(root):
        raise ValueError('Feature snapshot path escapes feature directory')
    if location.name != latest['snapshot_id']:
        raise ValueError('Feature snapshot pointer mismatch')
    return location


def _save_model(path: Path, model: Recommender) -> str:
    data={'user_ids':model.user_ids,'item_ids':model.item_ids,
          'popularity':model.popularity,'seen_data':model.seen.data,
          'seen_indices':model.seen.indices,'seen_indptr':model.seen.indptr,
          'seen_shape':np.asarray(model.seen.shape,dtype='int64')}
    if model.user_factors is not None:
        data.update(user_factors=model.user_factors,item_factors=model.item_factors)
    np.savez_compressed(path,**data)
    return sha256_file(path)


def load_model(model_dir: Path, kind: str) -> Recommender:
    """Load model only after checking manifest and model checksum; no pickle."""
    model_dir=Path(model_dir)
    manifest=json.loads((model_dir/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('model_version')!=MODEL_VERSION or kind not in manifest['artifacts']:
        raise ValueError('Invalid model version or kind')
    filename=f'{kind}.npz'
    if filename not in ('popularity.npz','collaborative_svd.npz'):
        raise ValueError('Unsupported model')
    path=model_dir/filename
    if sha256_file(path)!=manifest['artifacts'][kind]:
        raise ValueError('Model artifact checksum mismatch')
    with np.load(path,allow_pickle=False) as data:
        seen=sparse.csr_matrix((data['seen_data'],data['seen_indices'],data['seen_indptr']),
                               shape=tuple(map(int,data['seen_shape'])))
        uf=data['user_factors'].copy() if 'user_factors' in data else None
        itf=data['item_factors'].copy() if 'item_factors' in data else None
        return Recommender(kind,data['user_ids'].copy(),data['item_ids'].copy(),
                           data['popularity'].copy(),seen,uf,itf)


def _track_mlflow(model_dir: Path, report: dict, tracking_uri: str | None = None) -> dict:
    """Log two real experiment runs. Fail clearly if MLflow is unavailable."""
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError('MLflow is not installed. Install requirements.txt, or use --no-mlflow for offline testing.') from exc
    mlflow.set_tracking_uri(tracking_uri or 'sqlite:///mlflow.db')
    mlflow.set_experiment('dm4ml-movielens100k-model-training')
    run_ids={}
    for name in ('popularity','collaborative_svd'):
        with mlflow.start_run(run_name=f'{name}-{report["snapshot_id"]}') as run:
            mlflow.log_params({'model_type':name,'seed':report['seed'],
                               'requested_factors':report['n_factors'],
                               'actual_factors':report['actual_factors'],
                               'feature_snapshot':report['snapshot_id'],
                               'train_rows':report['train_interactions'],
                               'train_sha256':report['train_sha256']})
            # Training diagnostics, NOT held-out evaluation metrics.
            mlflow.log_metrics({'training_interactions':report['train_interactions'],
                                'training_users':report['users'],
                                'training_items':report['items']})
            mlflow.log_artifact(str(model_dir/f'{name}.npz'),artifact_path='model')
            mlflow.log_artifact(str(model_dir/'manifest.json'),artifact_path='model')
            run_ids[name]=run.info.run_id
    return run_ids


def train_models(prepared: Path, derived: Path, feature_root: Path,
                 model_dir: Path, report_dir: Path, n_factors: int=32, seed: int=42,
                 tracking: bool=True, tracking_uri: str | None=None) -> dict:
    prepared,derived,feature_root,model_dir,report_dir=map(Path,(prepared,derived,feature_root,model_dir,report_dir))
    location=_snapshot_path(feature_root)
    user_features,item_features,seen_pairs,feature_manifest=load_snapshot(location)
    expected_train=feature_manifest['source_sha256']['train_interactions']
    train_file=prepared/'train_interactions.csv'
    if sha256_file(train_file)!=expected_train:
        raise ValueError('Train CSV changed since feature snapshot; rerun milestone3')
    train=pd.read_csv(train_file)
    ui=pd.read_csv(derived/'user_index.csv').sort_values('row_index')
    ii=pd.read_csv(derived/'item_index.csv').sort_values('column_index')
    user_ids=ui.user_id.to_numpy(dtype='int64')
    item_ids=ii.item_id.to_numpy(dtype='int64')
    matrix_file=derived/'user_item_ratings_train.npz'
    transformation_report=json.loads((report_dir.parent/'transformation'/'transformation_report.json').read_text())
    if sha256_file(matrix_file)!=transformation_report['matrix']['sha256']:
        raise ValueError('Sparse rating matrix differs from transformation manifest')
    if expected_train!=transformation_report['train_csv_sha256']:
        raise ValueError('Transformation and feature snapshot train hashes differ')
    matrix=sparse.load_npz(matrix_file).tocsr()
    if matrix.nnz!=len(train) or len(seen_pairs)!=len(train):
        raise ValueError('Training interaction count does not match matrix/snapshot')
    if set(user_ids)!=set(user_features.user_id) or set(item_ids)!=set(item_features.item_id):
        raise ValueError('Training mapping and feature snapshot mismatch')
    # Verify matrix values against training events, not just counts.
    row_idx={int(v):i for i,v in enumerate(user_ids)}
    col_idx={int(v):i for i,v in enumerate(item_ids)}
    rows=train.user_id.map(row_idx).to_numpy()
    cols=train.item_id.map(col_idx).to_numpy()
    if np.any(~np.isclose(np.asarray(matrix[rows,cols]).ravel(),train.rating.to_numpy())):
        raise ValueError('Sparse training matrix values differ from prepared train file')
    baseline,personalized,fit_info=fit_models(matrix,user_ids,item_ids,n_factors,seed)
    model_dir.mkdir(parents=True,exist_ok=True)
    # Keep all committed model inputs/outputs separated from source inputs.
    artifacts={'popularity':_save_model(model_dir/'popularity.npz',baseline),
               'collaborative_svd':_save_model(model_dir/'collaborative_svd.npz',personalized)}
    manifest={'model_version':MODEL_VERSION,'snapshot_id':feature_manifest['snapshot_id'],
              'train_sha256':expected_train,'matrix_sha256':sha256_file(matrix_file),
              'artifacts':artifacts,'seed':seed,'n_factors':n_factors,
              'actual_factors':fit_info['actual_factors']}
    write_json(model_dir/'manifest.json',manifest)
    report={'stage':'model_training','status':'OK','models':list(artifacts),
            'model_dir':str(model_dir),'snapshot_id':manifest['snapshot_id'],
            'train_sha256':expected_train,'train_interactions':len(train),
            'users':len(user_ids),'items':len(item_ids),'seed':seed,
            'n_factors':n_factors,'actual_factors':fit_info['actual_factors'],
            'artifact_sha256':artifacts,'mlflow_enabled':tracking}
    if tracking:
        report['mlflow_run_ids']=_track_mlflow(model_dir,report,tracking_uri)
    else:
        report['mlflow_run_ids']={}
    report_dir.mkdir(parents=True,exist_ok=True)
    write_json(report_dir/'training_report.json',report)
    return report
