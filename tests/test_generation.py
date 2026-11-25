from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from fastapi import HTTPException
from test_generation_profiles import profile_dir
from generation_profiles import load_generation_profiles
from generation import generate, serialize_input


@pytest.mark.parametrize('content,finish,refusal,expected', [('{"label":"ok"}','stop',None,200), ('{}','stop',None,502), ('bad','stop',None,502), ('{"label":"ok"}','length',None,502), ('{"label":"ok"}','stop','refused',502)])
def test_generation(profile_dir, content, finish, refusal, expected):
    client = Mock()
    client.chat.completions.create.return_value = NS(choices=[NS(finish_reason=finish,message=NS(content=content,refusal=refusal))], model='gpt-4o-mini',usage=NS(prompt_tokens=10,completion_tokens=5))
    entry = load_generation_profiles(profile_dir)['demo']
    if expected == 200:
        result = generate(entry, serialize_input({'message':'test'}), client)
        assert result['result'] == {'label':'ok'}
        assert result['usage'] == {'input_tokens':10,'output_tokens':5,'embedding_tokens':0}
    else:
        with pytest.raises(HTTPException) as exc:
            generate(entry, '{}', client)
        assert exc.value.status_code == expected
    assert client.chat.completions.create.call_count == 1
    assert client.chat.completions.create.call_args.kwargs['response_format']['json_schema']['strict'] is True
