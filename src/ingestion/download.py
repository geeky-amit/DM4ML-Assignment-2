"""Download official MovieLens 100K and safely unpack required raw files."""
from __future__ import annotations
import hashlib
import ssl
from pathlib import Path
from urllib.request import Request, urlopen
import certifi
from zipfile import ZipFile, BadZipFile
from src.utils.io import write_json, sha256_file

URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
# Published SHA1 by Dive into Deep Learning dataset registry.
EXPECTED_SHA1 = 'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83'
FILES = ('u.data', 'u.user', 'u.item', 'u.genre', 'u.info', 'README')


def download_movielens(raw_root: Path, archive: Path | None = None, timeout: int = 60) -> dict:
    """Fetch and verify archive; never silently replace user-provided source data."""
    raw_root = Path(raw_root)
    target = raw_root / 'ml-100k'
    raw_root.mkdir(parents=True, exist_ok=True)
    if archive is None:
        archive = raw_root / 'ml-100k.zip'
        if not archive.exists():
            req = Request(URL, headers={'User-Agent': 'DM4ML-Milestone1/1.0'})
            try:
                context = ssl.create_default_context(cafile=certifi.where())
                with urlopen(req, timeout=timeout, context=context) as source, archive.open('wb') as sink:
                    import shutil
                    shutil.copyfileobj(source, sink)
            except Exception:
                archive.unlink(missing_ok=True)
                raise
    archive = Path(archive)
    if not archive.is_file():
        raise FileNotFoundError(f'Archive not found: {archive}')
    with archive.open('rb') as fh:
        digest = hashlib.file_digest(fh, 'sha1').hexdigest()
    if digest != EXPECTED_SHA1:
        raise ValueError(f'MovieLens archive SHA1 mismatch: {digest}; expected {EXPECTED_SHA1}. Do not use this archive.')
    target.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zf:
        names = set(zf.namelist())
        for filename in FILES:
            member = f'ml-100k/{filename}'
            if member not in names:
                raise ValueError(f'Archive missing required member: {member}')
            # Extract only our fixed allowlist. Prevent zip-slip/path traversal.
            with zf.open(member) as source, (target / filename).open('wb') as dest:
                import shutil
                shutil.copyfileobj(source, dest)
    manifest = {
        'source': URL, 'archive': str(archive), 'archive_sha1': digest,
        'extracted': {f: {'path': str(target / f), 'sha256': sha256_file(target / f)} for f in FILES},
    }
    write_json(raw_root / 'source_manifest.json', manifest)
    return manifest
