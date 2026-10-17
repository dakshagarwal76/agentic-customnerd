# Legacy runtime isolation

The legacy backend has process-global domain modules and a shared usage meter.
`run_agent` and `activate_domain` acquire the same reentrant process lock, so
startup and evaluation callers cannot bypass serialization. The HTTP worker
also holds it across meter reset, retries, execution and result construction.
Callers that manage their own global meter reset or result construction must
hold `isolated_legacy_request` around that whole unit too. A lock inside
`run_agent` cannot protect code executed by its caller outside that function.

`AGENT_RUNTIME_LOCK_TIMEOUT_SECONDS` defaults to 30 seconds and accepts a finite
number greater than zero and no more than 300. This is a local proposed default,
not a measured service-level target. A waiting call raises `LegacyRuntimeBusy`
without entering the protected operation. The domain HTTP endpoint returns 503;
the background worker emits a terminal error result so SSE clients do not wait
forever. Nested calls by the current owner are allowed.

This is only an acquisition deadline. It does not cancel a provider call or
release a lock held by a hung owner. Releasing another call's lock would allow
corruption. Provider deadlines and per-request backend/meter state are separate
work. Separate processes do not share this lock.

Offline tests stub pipelines and filesystem work. They demonstrate exclusion,
timeout handling and reentrancy, not provider reliability or production capacity.
