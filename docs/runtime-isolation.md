# Legacy runtime isolation

The backend serves queries with four worker threads, but the legacy pipeline and LLM meter use process-global state. A second worker can reset another request's telemetry, and a domain switch can replace prompts/retrieval modules during a query.

A shared process-local lock now covers the complete agent worker and the domain-switch mutation. The switch endpoint is synchronous so FastAPI runs its blocking lock wait in a worker instead of stopping the event loop. SSE delivery remains outside this lock.

This is a conservative serialization guard, not parallel request isolation. Queries wait for the preceding query; throughput is lower. It does not isolate separate server processes sharing the legacy files. Every caller that mutates those files must use the same lock. Health reads and direct evaluation helpers are not protected by the endpoint guard. There is no queue limit or cancellation mechanism added.

Offline tests cover two overlapping workers with distinct meter counts, lock release after an error, and AST wiring of the worker/domain endpoint. Twenty repeated before/after replays change from 0/20 to 20/20 separate meter totals. No model calls, production data, deployment or live scientific evaluation is involved.
