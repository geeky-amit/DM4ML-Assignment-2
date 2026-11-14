"""Load MovieLens raw files without coercing away malformed input values."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.utils.io import sha256_file, write_dataset, write_json

GENRES = ('unknown', 'action', 'adventure', 'animation', 'children', 'comedy',
          'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror',
          'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western')
USER_COLS = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
ITEM_COLS = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url', *GENRES]
INTERACTION_COLS = ['user_id', 'item_id', 'rating', 'timestamp']


def load_raw(folder: Path) -> dict[str, pd.DataFrame]:
    folder = Path(folder)
    for filename in ('u.user', 'u.item', 'u.data'):
        if not (folder / filename).is_file():
            raise FileNotFoundError(f'Missing {filename} in {folder}. Run download first.')
    # String dtype ensures invalid numeric fields remain inspectable for validation.
    users = pd.read_csv(folder / 'u.user', sep='|', names=USER_COLS,
                        header=None, dtype='string', keep_default_na=False,
                        encoding='latin-1', on_bad_lines='error')
    items = pd.read_csv(folder / 'u.item', sep='|', names=ITEM_COLS,
                        header=None, dtype='string', keep_default_na=False,
                        encoding='latin-1', on_bad_lines='error')
    interactions = pd.read_csv(folder / 'u.data', sep='\t', names=INTERACTION_COLS,
                               header=None, dtype='string', keep_default_na=False,
                               encoding='latin-1', on_bad_lines='error')
    return {'users': users, 'items': items, 'interactions': interactions}


def ingest(folder: Path, out: Path, parquet: bool = True) -> dict:
    folder, out = Path(folder), Path(out)
    tables = load_raw(folder)
    manifest = {'dataset': 'MovieLens 100K', 'source_dir': str(folder), 'tables': {}}
    for table, frame in tables.items():
        manifest['tables'][table] = {
            'rows': len(frame), 'columns': frame.columns.tolist(),
            'paths': write_dataset(frame, out / table, parquet=parquet),
        }
    manifest['input_sha256'] = {
        filename: sha256_file(folder / filename)
        for filename in ('u.user', 'u.item', 'u.data')
    }
    write_json(out / 'ingestion_manifest.json', manifest)
    return manifest
