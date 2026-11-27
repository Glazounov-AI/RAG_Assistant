# F5A — preliminary reliability patch

Request ID and monotonic budget propagated via headers. Content-free stage timings in uvicorn logs. Provider calls use remaining budget; SDK timeouts are not hard wall-clock deadlines. No automatic retries of user generation. RAG initialization retries every 30 seconds in background, readiness is global, not per profile. Cancellation of a worker thread does not stop its underlying synchronous operation. No production readiness or latency acceptance claimed. Live outage and recovery tests required. GigaChat unchanged.

## Final acceptance — 2026-11-27

- Python 3.14.4.
- RAG Assistant: 98 regression tests passed.
- AI IT Service Desk: 203 regression tests passed.
- Total: 301/301.
- Request ID propagation verified.
- Normal integration: 16/16 unique HTTP checks.
- Service Desk remains operational during RAG outage.
- Local escalation and ticket draft creation work without RAG.
- Provider fault simulated using an unreachable local OpenAI endpoint.
- HTTP API remains available: health 200, readiness 503.
- Background initialization retries verified.
- Provider fault requests: 2434, 2228 and 7 ms.
- Neutral user-facing fallback verified.
- Upstream HTTP 503 logging verified.
- RAG initialization recovered after restoring configuration
  and restarting the service.

### Remaining limitations

- Hard end-to-end deadline is not guaranteed.
- P95 <= 15 seconds is not established.
- Automatic recovery without restart remains unverified.
- HTTP 504 logging requires final runtime verification.
- Fallback clarification metadata may retain old wording.
- Targeted retrieval and token optimization remain in F5B.

Decision: GO to F5B.
