import pytest
from unittest.mock import Mock
from test_f2_regression import pipeline
from api import QueryRequest
from vector_store import VectorStore

@pytest.mark.parametrize('sources', [[], [''], ['../a.md'], ['a.md','a.md'], [' a.md'], ['x']*21])
def test_invalid_sources(sources):
    with pytest.raises(ValueError):
        QueryRequest(query='test', sources=sources)

def test_targeted_bypasses_cache_and_catalog(pipeline):
    pipeline.query('same')
    pipeline.knowledge_bases['support']['catalog'] = {'enabled': True}
    store = pipeline.vector_stores['support']
    store.search.side_effect = None
    store.search.return_value = [{'source':'a.md','text':'target','distance':0.1}]
    result = pipeline.query('same', sources=['a.md'])
    assert result['strategy'] == 'targeted' and not result['from_cache']
    store.search.assert_called_with('same', top_k=3, sources=['a.md'])
    assert result['retrieved_sources'][0]['source'] == 'a.md'
    pipeline.knowledge_bases['support'].pop('catalog')
    assert pipeline.query('same')['from_cache']

def test_no_context_no_generation(pipeline):
    store = pipeline.vector_stores['support']
    store.search.side_effect = None
    store.search.return_value = []
    result = pipeline.query('test', sources=['missing.md'])
    assert result['retrieved_sources'] == []
    assert result['model'] == 'local-no-context'
    pipeline.openai_client.chat.completions.create.assert_not_called()

def test_filter_applied_before_search():
    store = VectorStore.__new__(VectorStore)
    store.collection = Mock()
    store._create_embedding = Mock(return_value=[0.1])
    store.collection.get.return_value = {'ids':['a']}
    store.collection.query.return_value = {'ids':[['a']], 'documents':[['text']], 'distances':[[0.1]], 'metadatas':[[{'source':'a.md'}]]}
    assert store.search('test', sources=['a.md'])[0]['source'] == 'a.md'
    assert store.collection.query.call_args.kwargs['where'] == {'source':{'$in':['a.md']}}
    store.collection.query.return_value['metadatas'] = [[{'source':'other.md'}]]
    with pytest.raises(RuntimeError):
        store.search('test', sources=['a.md'])

def test_missing_source_skips_embedding():
    store = VectorStore.__new__(VectorStore)
    store.collection = Mock()
    store.collection.get.return_value = {'ids':[]}
    store._create_embedding = Mock()
    assert store.search('test', sources=['missing.md']) == []
    store._create_embedding.assert_not_called()
    store.collection.query.assert_not_called()
