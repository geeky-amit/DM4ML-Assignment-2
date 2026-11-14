"""Public CLI contract, including all earlier commands."""
import json
import subprocess
import sys
from pathlib import Path
from src.validation.run import validate_and_save
from src.preparation.run import prepare
from src.transformation.run import transform
from src.features.run import materialize_features


def test_train_and_recommend_cli(source_dir,tmp_path):
    root=tmp_path
    validate_and_save(source_dir,root/'data/validated',root/'data/errors',root/'reports/validation',parquet=False)
    prepare(root/'data/validated',root/'data/processed',root/'reports/preparation',train_ratio=.67,parquet=False)
    transform(root/'data/processed',root/'data/processed/derived',root/'reports/transformation',parquet=False)
    materialize_features(root/'data/processed',root/'data/features',root/'reports/features',parquet=False)
    repo=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'-m','scripts.pipeline','milestone4',
              '--root',str(root),'--factors','1','--no-mlflow'],cwd=repo,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'TRAIN OK:' in result.stdout
    query=subprocess.run([sys.executable,'-m','scripts.pipeline','recommend',
              '--root',str(root),'--user-id','2','--k','2'],cwd=repo,capture_output=True,text=True)
    assert query.returncode==0,query.stderr
    assert 'RECOMMEND popularity' in query.stdout
    assert 'RECOMMEND collaborative_svd' in query.stdout
