from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
from test_generation_profiles import profile_dir
from generation_profiles import load_generation_profiles
import api


@pytest.fixture
def http(monkeypatch, profile_dir):
    monkeypatch.setattr(api,'API_AUTH_ENABLED',False)
    monkeypatch.setattr(api,'GENERATION_ACCESS_KEY','test-generation-key')
    monkeypatch.setattr(api.app.state,'generation_profiles',load_generation_profiles(profile_dir),raising=False)
    monkeypatch.setattr(api.app.state,'pipeline',Mock(),raising=False)
    monkeypatch.setattr(api.app.state,'logger',Mock(),raising=False)
    monkeypatch.setattr(api.app.state,'rate_limiter',Mock(),raising=False)
    api.app.state.rate_limiter.check.return_value = True
    mock = Mock(return_value={'profile':'demo','result':{'label':'ok'},'model':'test','usage':{'embedding_tokens':0},'response_time_ms':1})
    monkeypatch.setattr(api,'generate',mock)
    with_client = TestClient(api.app)
    yield with_client, mock
    with_client.close()


def test_success(http):
    client, mock = http
    r = client.post('/generate',json={'profile':'demo','input':{'message':'test'}},headers={'X-Generation-Key':'test-generation-key'})
    assert r.status_code == 200
    assert mock.call_count == 1
    assert not api.app.state.pipeline.mock_calls
    assert api.app.state.rate_limiter.check.call_args.args[1] == 'generate:demo'
    assert api.app.state.logger.log.call_args.kwargs['query'] == '[redacted]'


@pytest.mark.parametrize('key,profile,body,expected',[('', 'demo',{},403),('test-generation-key','unknown',{},422),('test-generation-key','demo',{'text':'x'*9000},422)])
def test_rejected(http,key,profile,body,expected):
    client,mock=http
    assert client.post('/generate',json={'profile':profile,'input':body},headers={'X-Generation-Key':key}).status_code == expected
    mock.assert_not_called()


def test_limit(http):
    client,mock=http
    api.app.state.rate_limiter.check.return_value=False
    assert client.post('/generate',json={'profile':'demo','input':{}},headers={'X-Generation-Key':'test-generation-key'}).status_code == 429
    mock.assert_not_called()
