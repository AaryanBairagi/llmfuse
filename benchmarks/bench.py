"""Benchmark: sync vs async vs async + hedging, on simulated providers with a slow tail.

No API keys, no cost, and the same numbers on every run:

    uv run benchmarks/bench.py
"""

import asyncio
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor

from llmfuse import AsyncFuseClient, FuseClient, as_async

REQUESTS = 40
NORMAL = (0.05, 0.15)  # most calls take 50-150 ms...
SLOW = 1.0  # ...but some get stuck for 1 s: the "slow tail"
SLOW_CHANCE = 0.10  # 1 call in 10
HEDGE_AFTER = 0.3


class SimulatedProvider:
    """A sync provider whose delay depends only on (provider, prompt), so every run is identical."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        rng = random.Random(f"{self.name}:{prompt}")
        delay = SLOW if rng.random() < SLOW_CHANCE else rng.uniform(*NORMAL)
        time.sleep(delay)  # blocking, like a real HTTP call
        return "ok"


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: p95 = the value 95% of requests were faster than (or equal to)."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def make_providers() -> list[SimulatedProvider]:
    return [SimulatedProvider(name) for name in ("groq", "gemini", "claude")]


def run_sync() -> tuple[float, list[float], int]:
    providers = make_providers()
    client = FuseClient(providers=providers)
    latencies = []
    start = time.monotonic()
    for i in range(REQUESTS):
        began = time.monotonic()
        client.complete(f"q{i}")
        latencies.append(time.monotonic() - began)
    return time.monotonic() - start, latencies, sum(p.calls for p in providers)


async def run_async(hedge_after: float | None) -> tuple[float, list[float], int]:
    # Every in-flight call needs a worker thread; the default pool is only min(32, CPUs + 4).
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=2 * REQUESTS))
    providers = make_providers()
    client = AsyncFuseClient(providers=[as_async(p) for p in providers], hedge_after=hedge_after)

    async def one(i: int) -> float:
        began = time.monotonic()
        await client.complete(f"q{i}")
        return time.monotonic() - began

    start = time.monotonic()
    latencies = await asyncio.gather(*(one(i) for i in range(REQUESTS)))
    return time.monotonic() - start, list(latencies), sum(p.calls for p in providers)


def row(mode: str, result: tuple[float, list[float], int]) -> str:
    total, latencies, calls = result
    p50, p95, p99 = (percentile(latencies, p) for p in (50, 95, 99))
    return f"| {mode} | {total:.2f} s | {p50:.2f} s | {p95:.2f} s | {p99:.2f} s | {calls} |"


def main() -> None:
    print(f"{REQUESTS} requests · normal calls {NORMAL[0]}-{NORMAL[1]} s · {SLOW_CHANCE:.0%} stuck for {SLOW} s\n")
    print("| Mode | Total time | p50 | p95 | p99 | Provider calls |")
    print("|---|---|---|---|---|---|")
    print(row("`FuseClient` (sync, one after another)", run_sync()))
    print(row("`AsyncFuseClient` (all at once)", asyncio.run(run_async(None))))
    print(row(f"`AsyncFuseClient` + `hedge_after={HEDGE_AFTER}`", asyncio.run(run_async(HEDGE_AFTER))))


if __name__ == "__main__":
    main()