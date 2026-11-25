import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'assistant_api'))
from generation_profiles import load_generation_profiles


@pytest.fixture
def profile_dir(tmp_path):
    root = tmp_path / 'profiles'
    root.mkdir()
    (tmp_path / 'prompt.md').write_text('Extract a label.', encoding='utf-8')
    (tmp_path / 'schema.json').write_text(json.dumps({'type':'object','properties':{'label':{'type':'string'}},'required':['label'],'additionalProperties':False}))
    (root / 'demo.json').write_text(json.dumps({'profile':'demo','version':'v1','model':'gpt-4o-mini','prompt_file':'../prompt.md','schema_file':'../schema.json'}))
    return root


def test_disabled():
    assert load_generation_profiles('') == {}


def test_valid(profile_dir):
    p, prompt, schema = load_generation_profiles(profile_dir)['demo']
    assert p.log_content is False and prompt and schema['type'] == 'object'


def test_duplicate(profile_dir):
    (profile_dir / 'copy.json').write_bytes((profile_dir / 'demo.json').read_bytes())
    with pytest.raises(ValueError):
        load_generation_profiles(profile_dir)


def test_invalid_schema(profile_dir):
    (profile_dir.parent / 'schema.json').write_text('{"type":"invalid"}')
    with pytest.raises(ValueError):
        load_generation_profiles(profile_dir)


def test_remote_ref(profile_dir):
    (profile_dir.parent / 'schema.json').write_text('{"type":"object","$ref":"https://example.org/schema"}')
    with pytest.raises(ValueError):
        load_generation_profiles(profile_dir)
