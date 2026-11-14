import subprocess
import sys
from pathlib import Path


def test_milestone2_cli_from_milestone1_source(source_dir,tmp_path):
    raw=tmp_path/'data/raw/ml-100k'
    raw.mkdir(parents=True)
    for item in source_dir.iterdir():
        (raw/item.name).write_bytes(item.read_bytes())
    repo=Path(__file__).resolve().parents[1]
    for command in ('ingest','validate','milestone2'):
        completed=subprocess.run([sys.executable,'-m','scripts.pipeline',command,'--root',str(tmp_path),'--csv-only','--require-clean'],cwd=repo,capture_output=True,text=True)
        assert completed.returncode==0,completed.stderr
    assert (tmp_path/'data/processed/train_interactions.csv').exists()
    assert (tmp_path/'data/processed/derived/user_item_ratings_train.npz').exists()
    assert (tmp_path/'reports/preparation/preparation_report.json').exists()
