"""Bounded retrieval without completion, cache or catalog routing."""
from observability import stage
from vector_store import embedding_tokens
import config


def retrieve(pipeline, query, mode, sources=None, max_context_chars=12000):
    settings = pipeline.knowledge_bases[mode]
    token = embedding_tokens.set(None)
    try:
        with stage('retrieval'):
            docs = pipeline.vector_stores[mode].search(
                query, top_k=settings.get('top_k', config.TOP_K),
                **({'sources': sources} if sources is not None else {}))
        usage = embedding_tokens.get()
        chunks = []
        remaining = max_context_chars
        truncated = False
        for doc in docs:
            text = doc['text']
            if remaining <= 0:
                truncated = True
                break
            part = text[:remaining]
            truncated |= len(part) < len(text)
            chunks.append({'id': doc.get('id'), 'source': doc.get('source'),
                           'distance': doc.get('distance'), 'text': part})
            remaining -= len(part)
        return {'chunks': chunks, 'matched_chunks': len(docs),
                'context_chars': max_context_chars - remaining,
                'truncated': truncated,
                'strategy': 'targeted' if sources is not None else 'semantic',
                'usage': {'input_tokens': 0, 'output_tokens': 0, 'embedding_tokens': usage}}
    finally:
        embedding_tokens.reset(token)
