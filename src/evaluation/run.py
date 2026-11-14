"""Leakage-aware MovieLens top-K evaluation with defensible cohort diagnostics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation.metrics import ranking_metrics,genre_diversity
from src.features.run import load_snapshot
from src.training.run import _snapshot_path
from src.training.run import load_model
from src.utils.io import sha256_file,write_json

K_VALUES=(5,10)
MODELS=('popularity','collaborative_svd')


def _log_mlflow(metrics: dict, training_report: dict, tracking_uri: str | None) -> dict:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as e:
        raise RuntimeError('Install mlflow to track evaluation; or pass --no-mlflow') from e
    uri=tracking_uri or 'sqlite:///mlflow.db'
    mlflow.set_tracking_uri(uri)
    client=MlflowClient(tracking_uri=uri)
    run_ids=training_report.get('mlflow_run_ids',{})
    logged={}
    for model in MODELS:
        run_id=run_ids.get(model)
        if not run_id:
            raise ValueError(f'Missing MLflow training run ID for {model}. Rerun training with tracking enabled')
        # Check run exists before logging any results.
        client.get_run(run_id)
        for key,value in metrics[model].items():
            if isinstance(value,(float,int)) and not isinstance(value,bool) and np.isfinite(value):
                client.log_metric(run_id,f'eval_{key}',float(value))
        logged[model]=run_id
    return logged


def _save_chart(table: pd.DataFrame, chart_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=table[table.metric.isin(['precision','recall','ndcg','coverage','diversity'])]
    fig,axes=plt.subplots(2,5,figsize=(17,6.5),sharey='col')
    for i,k in enumerate(K_VALUES):
        for j,metric in enumerate(('precision','recall','ndcg','coverage','diversity')):
            ax=axes[i,j]
            part=rows[(rows.k==k)&(rows.metric==metric)].set_index('model')
            vals=[float(part.loc[m,'value']) for m in MODELS]
            ax.bar(['Popularity','SVD'],vals)
            ax.set_ylim(0,1)
            ax.set_title(f'{metric.title()}@{k}')
            ax.tick_params(axis='x',labelrotation=20,labelsize=8)
            for b,v in zip(ax.patches,vals):
                ax.text(b.get_x()+b.get_width()/2,v+0.015,f'{v:.3f}',ha='center',fontsize=8)
    fig.suptitle('MovieLens 100K | Held-out ranking evaluation (actual computed metrics)')
    fig.tight_layout()
    chart_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(chart_path,dpi=160,bbox_inches='tight')
    plt.close(fig)


def evaluate_models(prepared: Path, features: Path, model_dir: Path, training_report_file: Path,
                    report_dir: Path, relevant_rating: int=4, k_values=K_VALUES,
                    tracking: bool=True,tracking_uri: str | None=None) -> dict:
    prepared,features,model_dir,training_report_file,report_dir=map(Path,(prepared,features,model_dir,training_report_file,report_dir))
    if tuple(k_values)!=K_VALUES:
        raise ValueError('This comparison requires the fixed K values (5,10)')
    if not isinstance(relevant_rating,int) or not 1<=relevant_rating<=5:
        raise ValueError('relevant_rating must be an integer between 1 and 5')
    train_file=prepared/'train_interactions.csv'
    test_file=prepared/'test_interactions.csv'
    train=pd.read_csv(train_file)
    test=pd.read_csv(test_file)
    required={'user_id','item_id','rating','timestamp'}
    if not required.issubset(train) or not required.issubset(test):
        raise ValueError('Train/test file lacks required columns')
    if train.empty or test.empty or train.timestamp.max()>=test.timestamp.min():
        raise ValueError('Evaluation requires nonempty strictly chronological train/test')
    for df in (train,test):
        if df[list(required)].isna().any().any():
            raise ValueError('Nulls in interactions')
        if df[['user_id','item_id']].duplicated().any():
            raise ValueError('Duplicate interaction keys')
        if not df.rating.between(1,5).all():
            raise ValueError('Invalid rating')
    training_report=json.loads(training_report_file.read_text(encoding='utf-8'))
    model_manifest=json.loads((model_dir/'manifest.json').read_text(encoding='utf-8'))
    if (sha256_file(train_file)!=training_report['train_sha256'] or
        training_report['train_sha256']!=model_manifest['train_sha256'] or
        training_report['snapshot_id']!=model_manifest['snapshot_id']):
        raise ValueError('Training data / model provenance mismatch')
    snapshot=_snapshot_path(features)
    _,item_features,seen_pairs,feature_manifest=load_snapshot(snapshot)
    if feature_manifest['snapshot_id']!=model_manifest['snapshot_id']:
        raise ValueError('Feature snapshot differs from trained models')
    if sha256_file(train_file)!=feature_manifest['source_sha256']['train_interactions']:
        raise ValueError('Training input changed after feature snapshot')
    models={m:load_model(model_dir,m) for m in MODELS}
    ids=set(map(int,models['popularity'].item_ids))
    users=set(map(int,models['popularity'].user_ids))
    if ids != set(map(int,item_features.item_id)):
        raise ValueError('Model and feature item universes disagree')
    for model in models.values():
        if set(map(int,model.item_ids))!=ids or set(map(int,model.user_ids))!=users:
            raise ValueError('Model candidate universes differ')
    if len(seen_pairs)!=len(train):
        raise ValueError('Seen-pair snapshot count does not match train')
    history=set(zip(map(int,train.user_id),map(int,train.item_id)))
    if history!=set(zip(map(int,seen_pairs.user_id),map(int,seen_pairs.item_id))):
        raise ValueError('Seen-pair snapshot does not match training history')
    relevant=test.loc[test.rating>=relevant_rating,['user_id','item_id']]
    all_relevant=len(relevant)
    # Unseen relevant interactions are evaluated; earlier-seen duplicates are not rankable.
    eligible_relevant={}
    unrankable=0
    for row in relevant.itertuples(index=False):
        pair=(int(row.user_id),int(row.item_id))
        if pair in history or pair[1] not in ids:
            unrankable+=1
            continue
        eligible_relevant.setdefault(pair[0],set()).add(pair[1])
    if not eligible_relevant:
        raise ValueError('No eligible held-out relevant user/item pairs')
    genre_cols=[c for c in item_features.columns if c not in ('item_id','item_popularity','item_avg_rating','item_genres')]
    genres={int(row.item_id):frozenset(g for g in str(row.item_genres).split('|') if g and g!='nan')
            for row in item_features.itertuples(index=False)}
    # Genre masks are source attributes; here we use the pre-cutoff feature snapshot only.
    results={}
    comparison=[]
    recommendation_rows=[]
    for name,model in models.items():
        sums={k:{'precision':0.,'recall':0.,'ndcg':0.,'diversity':0.} for k in K_VALUES}
        unions={k:set() for k in K_VALUES}
        fallback_count=0
        for user_id,positive in sorted(eligible_relevant.items()):
            recs=model.recommend(user_id,max(K_VALUES))
            recommended=[int(r['item_id']) for r in recs]
            if any((user_id,i) in history for i in recommended):
                raise ValueError('Model recommended a training-seen item')
            if len(recommended)!=len(set(recommended)) or any(i not in ids for i in recommended):
                raise ValueError('Invalid recommendation candidate')
            if name=='collaborative_svd' and recs and recs[0]['fallback']:
                fallback_count+=1
            for k in K_VALUES:
                top=recommended[:k]
                score=ranking_metrics(top,positive,k)
                score['diversity']=genre_diversity(top,genres)
                for m,value in score.items():
                    sums[k][m]+=value
                unions[k].update(top)
                for rank,item in enumerate(top,1):
                    recommendation_rows.append({'model':name,'user_id':user_id,'k':k,'rank':rank,
                        'item_id':item,'relevant':item in positive})
        metrics={'eligible_users':len(eligible_relevant),'cold_start_fallback_users':fallback_count}
        for k in K_VALUES:
            for metric in ('precision','recall','ndcg','diversity'):
                val=sums[k][metric]/len(eligible_relevant)
                metrics[f'{metric}_at_{k}']=val
                comparison.append({'model':name,'k':k,'metric':metric,'value':val})
            coverage=len(unions[k])/len(ids)
            metrics[f'coverage_at_{k}']=coverage
            comparison.append({'model':name,'k':k,'metric':'coverage','value':coverage})
        results[name]=metrics
    report_dir.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(comparison)
    table.to_csv(report_dir/'model_comparison.csv',index=False)
    pd.DataFrame(recommendation_rows).to_csv(report_dir/'recommendation_audit.csv',index=False)
    _save_chart(table,report_dir/'model_comparison.png')
    report={'stage':'model_evaluation','status':'OK','models':MODELS,
            'k_values':K_VALUES,'relevance_threshold':relevant_rating,
            'relevance_policy':'test rating >= threshold; exclude previously seen training pairs',
            'cohort_policy':'users with at least one unseen relevant held-out movie; all known items minus training history are candidates',
            'precision_denominator':'K even if fewer than K candidates exist',
            'aggregation':'macro mean over eligible users; coverage on global known item catalog',
            'diversity':'mean pairwise genre Jaccard distance; empty genre unions score zero',
            'eligible_users':len(eligible_relevant),'all_test_interactions':len(test),
            'test_relevant_interactions':all_relevant,'unrankable_relevant_interactions':unrankable,
            'test_sha256':sha256_file(test_file),'train_sha256':sha256_file(train_file),
            'snapshot_id':model_manifest['snapshot_id'],'metrics':results,
            'artifacts':{'comparison_csv':'model_comparison.csv','chart':'model_comparison.png','recommendation_audit':'recommendation_audit.csv'},
            'mlflow_enabled':tracking}
    if tracking:
        report['mlflow_run_ids']=_log_mlflow(results,training_report,tracking_uri)
    else:
        report['mlflow_run_ids']={}
    write_json(report_dir/'evaluation_report.json',report)
    return report
