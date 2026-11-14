import pandas as pd
import pytest
from src.ingestion.load import load_raw
from src.validation.rules import validate
from src.validation.run import validate_and_save


def test_clean_fixture(source_dir):
    result = validate(load_raw(source_dir))
    assert result.report['status']=='PASS'
    assert result.report['total_rejected_rows']==0
    assert len(result.accepted['interactions'])==3


def test_invalid_rating_timestamp_ids_and_duplicate(source_dir):
    tables = load_raw(source_dir)
    bad = tables['interactions'].iloc[0].copy()
    bad['rating'] = '6'
    bad['timestamp'] = 'garbage'
    bad['user_id'] = '999'
    bad['item_id'] = '888'
    tables['interactions'] = pd.concat([tables['interactions'],pd.DataFrame([bad])], ignore_index=True)
    # Duplicate a valid interaction as another row.
    tables['interactions'] = pd.concat([tables['interactions'],tables['interactions'].iloc[[0]]],ignore_index=True)
    result = validate(tables)
    assert len(result.rejected['interactions'])==2
    reasons = '|'.join(result.rejected['interactions'].error_reasons)
    for reason in ('invalid_rating_type_or_range','invalid_timestamp','unknown_user_id',
                   'unknown_item_id','duplicate_user_item_interaction'):
        assert reason in reasons
    assert len(result.accepted['interactions'])==3


def test_invalid_user_and_item_cascade_foreign_keys(source_dir):
    tables = load_raw(source_dir)
    tables['users'].loc[0,'age'] = 'bad'
    tables['items'].loc[1,'action'] = 'banana'
    result = validate(tables)
    assert len(result.rejected['users'])==1
    assert len(result.rejected['items'])==1
    assert len(result.rejected['interactions'])==2
    assert result.report['rules']['interactions.unknown_user_id']==2
    assert result.report['rules']['interactions.unknown_item_id']==1


def test_required_fields_and_schema(source_dir):
    tables = load_raw(source_dir)
    tables['users'].loc[1, 'occupation'] = ''
    tables['items'].loc[0, 'title'] = ''
    result = validate(tables)
    assert len(result.rejected['users'])==1
    assert len(result.rejected['items'])==1
    del tables['interactions']['rating']
    with pytest.raises(ValueError,match='missing required columns'):
        validate(tables)


def test_invalid_types_date_and_ages(source_dir):
    tables=load_raw(source_dir)
    tables['users'].loc[0,'user_id']='1.0'
    tables['users'].loc[1,'age']='200'
    tables['items'].loc[0,'release_date']='someday'
    tables['interactions'].loc[0,'rating']='3.5'
    tables['interactions'].loc[1,'timestamp']='999999999999999999'
    result=validate(tables)
    assert len(result.rejected['users'])==2
    assert len(result.rejected['items'])==1
    assert len(result.rejected['interactions'])>=2
    assert result.report['rules']['interactions.invalid_timestamp']==1


def test_write_quarantine_and_report(source_dir,tmp_path):
    with (source_dir/'u.data').open('a') as f:
        f.write('4\t2\t9\t-99999\n')
    report=validate_and_save(source_dir,tmp_path/'valid',tmp_path/'errors',tmp_path/'reports',parquet=False)
    assert report['status']=='PASS_WITH_QUARANTINE'
    assert report['tables']['interactions']['rejected_rows']==1
    assert (tmp_path/'errors/interactions_rejected.csv').exists()
    assert (tmp_path/'reports/validation_report.json').exists()
    assert len(pd.read_csv(tmp_path/'valid/interactions.csv'))==3


def test_missing_release_date_is_allowed_and_preserves_references(source_dir):
    """A missing movie release date is optional, not a broken foreign key."""
    tables = load_raw(source_dir)
    movie_id = tables['items'].iloc[1]['item_id']
    tables['items'].loc[1, 'release_date'] = ''
    result = validate(tables)
    assert result.report['status'] == 'PASS'
    assert result.report['total_rejected_rows'] == 0
    assert movie_id in set(result.accepted['items']['item_id'])
    assert movie_id in set(result.accepted['interactions']['item_id'])
    assert result.report['rules'].get('items.missing_release_date', 0) == 0
