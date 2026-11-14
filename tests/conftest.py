from pathlib import Path
import pandas as pd
import pytest
from src.ingestion.load import ITEM_COLS

@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    folder = tmp_path / 'ml-100k'
    folder.mkdir()
    (folder/'u.user').write_text('1|25|M|student|12345\n2|30|F|engineer|54321\n3|18|M|artist|11111\n', encoding='latin-1')
    genre = '|'.join(['1'] + ['0']*18)
    (folder/'u.item').write_text(
        f'1|Movie Alpha|01-Jan-1997||http://example.test/a|{genre}\n'
        f'2|Movie Beta|01-Jan-1997||http://example.test/b|{genre}\n', encoding='latin-1')
    (folder/'u.data').write_text('1\t1\t5\t875000000\n1\t2\t4\t875000001\n2\t1\t3\t875000002\n',encoding='utf-8')
    return folder
