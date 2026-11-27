# F5B.1 — Retrieval-only API

POST /retrieve accepts query, mode, optional sources (same validation as /query), and max_context_chars (1–24000, default 12000). Profile top_k is retained. Returns chunks with source, id, distance and text, matched_chunks (before context truncation), context_chars, truncated, strategy and usage. No completion, catalog routing or answer cache. Unknown embedding usage is null; missing targeted sources report zero. Distances are not calibrated confidence scores. Character limit is not a token limit.

Raw KB access requires configured RAG_API_ACCESS_KEY and X-API-Key even if legacy API authentication is disabled. Shares mode rate limit with /query. Logs content redacted. No per-profile authorization yet; do not expose publicly before F5C. Existing /query and /generate contracts unchanged. Provider failure and deadline limitations remain as in F5A.

F5B.2 will consume this API and compare end-to-end costs and quality. No measured savings claimed yet.

## Integration acceptance — 2026-11-27

- Python 3.14.4.
- Regression: 108/108 passed.
- Integration: 12/12 passed.
- Request ID propagation: 12/12.
- Retrieval authentication verified.
- Semantic retrieval parity verified.
- Targeted retrieval: KB-011 and KB-007 verified.
- Missing source: zero token consumption.
- Context limit and truncation verified.
- Structured registry compatibility verified.
- Generation calls during retrieval: zero.

### Comparative measurements

| Scenario | Query ms | Retrieve ms | Query tokens | Retrieve tokens |
|---|---:|---:|---:|---:|
| VPN semantic | 17118 | 3909 | 3484 | 24 |
| VPN targeted | 7133 | 687 | 3174 | 24 |
| Outlook targeted | 6444 | 857 | 3168 | 20 |

Total tokens: 9826 -> 68 (-99.31%).
Mean latency: 10232 -> 1818 ms (-82.2%).

### Limitations

- Single measurement per operation.
- End-to-end Service Desk savings not yet established.
- Production P95 target remains unverified.
- D34 and D35 require Service Desk integration.

Decision: GO to F5B.2.
