"""Structured generation without retrieval, embeddings or caching."""
import json
import time
from observability import remaining

from jsonschema import Draft202012Validator
from openai import OpenAI, APITimeoutError, APIConnectionError, APIStatusError
from fastapi import HTTPException


def serialize_input(value):
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    except (ValueError, TypeError, RecursionError):
        raise HTTPException(422, 'Invalid generation input') from None


def generate(entry, text, client=None):
    profile, prompt, schema = entry
    started = time.perf_counter()
    try:
        if client is None:
            client = OpenAI(max_retries=0, timeout=profile.timeout_seconds)
            with client:
                return generate(entry, text, client)
        response = client.chat.completions.create(
            model=profile.model,
            messages=[{'role': 'system', 'content': prompt}, {'role': 'user', 'content': text}],
            response_format={'type': 'json_schema', 'json_schema': {'name': profile.profile, 'strict': True, 'schema': schema}},
            temperature=profile.temperature,
            max_completion_tokens=profile.max_tokens,
            timeout=remaining(profile.timeout_seconds, reserve=0.25),
        )
    except (APITimeoutError, TimeoutError):
        raise HTTPException(504, 'Generation timed out') from None
    except APIConnectionError:
        raise HTTPException(503, 'Generation provider unavailable') from None
    except APIStatusError as exc:
        raise HTTPException(503 if exc.status_code == 429 or exc.status_code >= 500 else 502, 'Generation provider error') from None
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, 'Generation unavailable') from None
    try:
        choice = response.choices[0]
        if choice.finish_reason != 'stop' or choice.message.refusal:
            raise ValueError('Incomplete or refused')
        result = json.loads(choice.message.content, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Invalid constant')))
        Draft202012Validator(schema).validate(result)
        if not isinstance(result, dict):
            raise ValueError('Expected object')
        usage = {'input_tokens': response.usage.prompt_tokens, 'output_tokens': response.usage.completion_tokens, 'embedding_tokens': 0} if response.usage else {'input_tokens': None, 'output_tokens': None, 'embedding_tokens': 0}
    except Exception:
        raise HTTPException(502, 'Invalid structured generation result') from None
    return {'profile': profile.profile, 'result': result, 'model': response.model, 'usage': usage, 'response_time_ms': (time.perf_counter()-started)*1000}
