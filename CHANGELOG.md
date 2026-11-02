# Changelog

All notable changes to llmfuse are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-02

### Added

- `AsyncFuseClient`: the same retries, circuit breakers, rate limits and failover as `FuseClient`,
  with `await client.complete(...)`, so many requests can run at once.
- `as_async(provider)` / `AsyncAdapter`: use any sync provider with the async client. The blocking
  HTTP call runs in a worker thread, so the event loop never blocks.
- `AsyncProvider` protocol for native async providers, and `async_retry_call`, the async twin of `retry_call`.
- Deadlines: `timeout=` caps a whole call (every retry, wait and provider) and raises the new
  `DeadlineExceededError`. It can be set per client or per call.
- Least-busy routing: `routing="least_busy"` sends overlapping requests to the provider with the
  fewest calls in flight. `in_flight(name)` shows the current count.
- Hedged requests: `hedge_after=` starts a backup request on the next provider when the first one
  is slow. The first good answer wins and the others are cancelled.
- `FakeAsyncProvider` in `llmfuse.testing`.
- `benchmarks/bench.py`: sync vs async vs hedged, on simulated providers with a slow tail.
- Examples: `check_providers_async.py`, `check_routing.py`, `check_deadline.py`, `check_hedging.py`.

### Changed

- `FuseClient` and `AsyncFuseClient` share their setup code (`build_guards`). `FuseClient`
  behaves exactly as before.

## [0.1.0] - 2026-10-07

### Added

- `FuseClient` with retries (exponential backoff, full jitter, `Retry-After`), per-provider
  circuit breakers, per-provider token-bucket rate limiting and multi-provider failover.
- Providers: Groq, Gemini, OpenAI, Anthropic and any OpenAI-compatible API.
- `llmfuse.testing`: fake providers, clocks and transports for offline tests.

[0.2.0]: https://github.com/AaryanBairagi/llmfuse/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/AaryanBairagi/llmfuse/releases/tag/v0.1.0