import subprocess
import sys
from pathlib import Path


def test_cli_ingest_and_validation(source_dir,tmp_path):
    raw = tmp_path/'data/raw/ml-100k'
    raw.mkdir(parents=True)
    for file in source_dir.iterdir():
        (raw/file.name).write_bytes(file.read_bytes())
    root=Path(__file__).resolve().parents[1]
    for cmd in ('ingest','validate'):
        process=subprocess.run([sys.executable,'-m','scripts.pipeline',cmd,'--root',str(tmp_path),'--csv-only'],
                               cwd=root,capture_output=True,text=True)
        assert process.returncode==0,process.stderr
    assert (tmp_path/'reports/validation/validation_report.json').exists()
