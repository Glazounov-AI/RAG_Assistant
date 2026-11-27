"""Request-scoped deadline and content-free telemetry."""
import logging
import re
import time
import uuid
from contextvars import ContextVar
from contextlib import contextmanager

request_id = ContextVar('request_id', default='')
deadline = ContextVar('deadline', default=None)
log = logging.getLogger('uvicorn.error')

def remaining(cap, reserve=0.0):
    end = deadline.get()
    value = min(cap, end-time.monotonic()-reserve) if end is not None else cap
    if value <= 0:
        raise TimeoutError('Request budget exhausted')
    return value

@contextmanager
def stage(name):
    start = time.monotonic()
    outcome = 'ok'
    try:
        yield
    except BaseException as exc:
        outcome = type(exc).__name__
        raise
    finally:
        log.info('request_id=%s stage=%s elapsed_ms=%.1f outcome=%s', request_id.get(), name, (time.monotonic()-start)*1000, outcome)

def install(app, accept_budget=False):
    @app.middleware('http')
    async def trace(request, call_next):
        supplied = request.headers.get('X-Request-ID', '')
        rid = supplied if re.fullmatch(r'[a-f0-9]{32}', supplied) else uuid.uuid4().hex
        budget = 28.0
        if accept_budget:
            try:
                raw = int(request.headers.get('X-Request-Budget-Ms', '28000'))
                budget = max(0.001, min(28.0, raw/1000))
            except ValueError:
                pass
        a = request_id.set(rid)
        b = deadline.set(time.monotonic()+budget)
        start = time.monotonic()
        try:
            response = await call_next(request)
            response.headers['X-Request-ID'] = rid
            log.info('request_id=%s stage=http status=%s elapsed_ms=%.1f', rid, response.status_code, (time.monotonic()-start)*1000)
            return response
        finally:
            deadline.reset(b)
            request_id.reset(a)
