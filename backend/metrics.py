"""Benchmark metric calculations kept independent from the HTTP runner."""
from __future__ import annotations

import math
import statistics
from typing import Iterable, Mapping, Any


def percentile(values: Iterable[float], p: float) -> float | None:
    values = sorted(float(v) for v in values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    x = (len(values) - 1) * p
    low, high = math.floor(x), math.ceil(x)
    return values[low] if low == high else values[low] * (high - x) + values[high] * (x - low)


def batch_metrics(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if r.get("ok")]
    started = [float(r["started_s"]) for r in ok]
    finished = [float(r["finished_s"]) for r in ok]
    first = [float(r["first_token_s"]) for r in ok if r.get("first_token_s") is not None]
    ttft = [float(r["ttft_s"]) for r in ok if r.get("ttft_s") is not None]
    stream = [float(r["stream_decode_tps"]) for r in ok if r.get("stream_decode_tps") is not None]
    total = sum(int(r.get("completion_tokens") or 0) for r in ok)
    wall = max(1e-9, max(finished) - min(started)) if started else 0.0
    decode_window = max(1e-9, max(finished) - min(first)) if first and finished else 0.0
    # Exclude the first token from decode work: its arrival is already represented by TTFT.
    decode_tokens = sum(max(0, int(r.get("completion_tokens") or 0) - 1) for r in ok)
    return {
        "ok_requests": len(ok),
        "failed_requests": len(results) - len(ok),
        "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in ok),
        "completion_tokens": total,
        "wall_s": wall,
        "aggregate_e2e_tps": total / wall if wall and total else 0.0,
        "aggregate_decode_tps": decode_tokens / decode_window if decode_window and decode_tokens else 0.0,
        "avg_stream_decode_tps": statistics.mean(stream) if stream else None,
        "ttft_avg_s": statistics.mean(ttft) if ttft else None,
        "ttft_p50_s": percentile(ttft, 0.5),
        "ttft_p95_s": percentile(ttft, 0.95),
        "ttft_max_s": max(ttft) if ttft else None,
        "finish_reasons": [r.get("finish_reason") for r in ok if r.get("finish_reason") is not None],
        "errors": [r.get("error") for r in results if not r.get("ok") and r.get("error")],
    }


def sweet_spot(rows: list[Mapping[str, Any]], stream_ratio: float = 0.70, ttft_multiplier: float = 3.0, ttft_add: float = 1.0) -> dict[str, Any] | None:
    """Return the best balanced concurrency using the first successful row as baseline."""
    ordered = sorted((r for r in rows if not r.get("failed_requests")), key=lambda r: int(r["concurrency"]))
    if not ordered:
        return None
    baseline = ordered[0]
    base_stream = float(baseline.get("avg_stream_decode_tps") or 0.0)
    base_ttft = float(baseline.get("ttft_p50_s") or 0.0)
    eligible = []
    for row in ordered:
        stream = float(row.get("avg_stream_decode_tps") or 0.0)
        ttft = float(row.get("ttft_p50_s") or 0.0)
        if (not base_stream or stream >= base_stream * stream_ratio) and ttft <= max(base_ttft * ttft_multiplier, base_ttft + ttft_add):
            eligible.append(row)
    if not eligible:
        eligible = [baseline]
    chosen = max(eligible, key=lambda r: float(r.get("aggregate_decode_tps") or r.get("aggregate_e2e_tps") or 0.0))
    return {
        "concurrency": chosen["concurrency"],
        "reason": "highest aggregate throughput within per-stream and TTFT guardrails",
        "stream_ratio": stream_ratio,
        "ttft_multiplier": ttft_multiplier,
        "ttft_add_s": ttft_add,
        "baseline_concurrency": baseline["concurrency"],
    }
