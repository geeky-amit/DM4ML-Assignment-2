import json
import pytest
from src.ingestion.load import load_raw, ingest
from src.ingestion.download import download_movielens


def test_load_and_ingest_manifest(source_dir, tmp_path):
    tables = load_raw(source_dir)
    assert {k:len(v) for k,v in tables.items()} == {'users':3,'items':2,'interactions':3}
    assert len(tables['items'].columns) == 24
    manifest = ingest(source_dir, tmp_path/'staged', parquet=False)
    assert (tmp_path/'staged/users.csv').exists()
    assert len(manifest['input_sha256']['u.data']) == 64
    assert json.loads((tmp_path/'staged/ingestion_manifest.json').read_text())['tables']['interactions']['rows'] == 3


def test_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path)


def test_bad_archive_rejected(tmp_path):
    archive = tmp_path/'fake.zip'
    archive.write_bytes(b'not an actual official archive')
    with pytest.raises(ValueError, match='SHA1 mismatch'):
        download_movielens(tmp_path/'raw', archive=archive)
