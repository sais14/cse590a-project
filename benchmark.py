"""benchmark.py — Run one config (A/B/C) over the 30 patients; dump metrics.

Configs (build_plan.md 5):
  A  single-shot baseline   (agent.run_single_shot; one LLM call per patient)
  B  full agent, prefix cache OFF
  C  full agent, prefix cache ON

B and C share identical client code (agent.run_agent). The ONLY difference is a
vLLM *server* flag, so this script does NOT manage the server. The operator must
(re)launch vLLM with the appropriate flag before each run:

    # Config B (cache OFF):
    python -m vllm.entrypoints.openai.api_server --model <MODEL> \
        --dtype bfloat16 --max-model-len 8192 --port 8000 \
        --gpu-memory-utilization 0.9 --no-enable-prefix-caching
    python benchmark.py --config B

    # Config C (cache ON) — restart the server first:
    python -m vllm.entrypoints.openai.api_server --model <MODEL> \
        --dtype bfloat16 --max-model-len 8192 --port 8000 \
        --gpu-memory-utilization 0.9 --enable-prefix-caching
    python benchmark.py --config C

Each invocation writes results/raw_<config>.jsonl (one line per patient) and
recomputes results/aggregate.json across whatever raw_*.jsonl exist.

Local wiring smoke test (no GPU/server):
    python benchmark.py --config B --mock --no-metrics --limit 3 --output-dir /tmp/bt
"""
import argparse
import asyncio
import json
import os
import statistics as stats

import agent
import tools

# vLLM Prometheus metric names we care about (build_plan.md 4.3).
VLLM_METRICS = [
    "vllm:gpu_prefix_cache_hit_rate",
    "vllm:gpu_cache_usage_perc",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
]
# Counters get a before/after delta; gauges are read as snapshots.
COUNTER_METRICS = {"vllm:prompt_tokens_total", "vllm:generation_tokens_total"}


def metrics_url_from_base(base_url):
    # base_url like http://localhost:8000/v1 -> http://localhost:8000/metrics
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root.rstrip("/") + "/metrics"


def scrape_vllm_metrics(metrics_url):
    """Return {metric_name: float} summed across label series. None on failure."""
    try:
        import requests
        text = requests.get(metrics_url, timeout=5).text
    except Exception as exc:  # noqa: BLE001
        return {"_error": str(exc)}
    out = {}
    for name in VLLM_METRICS:
        total, found = 0.0, False
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if line.startswith(name + " ") or line.startswith(name + "{"):
                try:
                    total += float(line.rsplit(" ", 1)[1])
                    found = True
                except (ValueError, IndexError):
                    pass
        out[name] = total if found else None
    return out


def metrics_delta(before, after):
    """Counters -> after-before delta; gauges -> after snapshot."""
    if not before or not after or "_error" in before or "_error" in after:
        return {}
    delta = {}
    for name in VLLM_METRICS:
        b, a = before.get(name), after.get(name)
        if a is None:
            delta[name] = None
        elif name in COUNTER_METRICS and b is not None:
            delta[name] = a - b
        else:
            delta[name] = a
    return delta


def summarize_worker_records(records):
    """Group CallRecords by worker into per-worker latency/ttft/tokens/validity."""
    per = {}
    for r in records:
        w = r.worker
        d = per.setdefault(w, {
            "n_calls": 0, "latency_s": 0.0, "ttft_s": None,
            "prompt_tokens": 0, "completion_tokens": 0,
            "final_attempt": None, "json_valid": None, "schema_valid": None,
            "had_correction": False,
        })
        d["n_calls"] += 1
        d["latency_s"] += r.latency_s
        if d["ttft_s"] is None:
            d["ttft_s"] = r.ttft_s          # first call's TTFT
        d["prompt_tokens"] = r.prompt_tokens  # last call's prompt size
        d["completion_tokens"] += r.completion_tokens
        d["final_attempt"] = r.attempt
        d["json_valid"] = r.json_valid
        d["schema_valid"] = r.schema_valid
        if r.is_correction:
            d["had_correction"] = True
    return per


async def run_one_patient(client, config, entry, model, metrics_url, use_metrics):
    pid = entry["patient_id"]
    note = open(entry["path"], encoding="utf-8").read()
    tool_results = tools.build_tool_results(pid)  # deterministic, computed once

    before = scrape_vllm_metrics(metrics_url) if use_metrics else None
    if config == "A":
        res = await agent.run_single_shot(
            client, pid, note, tool_results=tool_results, model=model, stream=True)
    else:  # B or C — identical code; the server flag differs
        res = await agent.run_agent(
            client, pid, note, tool_results=tool_results, model=model,
            use_guided=True, reflect=True, stream=True)
    after = scrape_vllm_metrics(metrics_url) if use_metrics else None

    recs = res.records
    total_ptok = sum(r.prompt_tokens for r in recs)
    total_ctok = sum(r.completion_tokens for r in recs)
    n_json_valid = sum(1 for r in recs if r.json_valid)
    n_schema_valid = sum(1 for r in recs if r.schema_valid)

    record = {
        "config": config,
        "note_id": entry["note_id"],
        "patient_id": pid,
        "note_chars": entry["chars"],
        "note_tokens": entry["approx_tokens"],
        "wall_clock_s": res.wall_clock_s,
        "n_llm_calls": len(recs),
        "total_prompt_tokens": total_ptok,
        "total_completion_tokens": total_ctok,
        "per_worker": summarize_worker_records(recs),
        "reflection": {
            "findings": res.reflection.get("findings", []),
            "offending": res.reflection.get("offending", []),
            "corrected": res.reflection.get("corrected", []),
            "n_triggered": len(res.reflection.get("corrected", [])),
        },
        "validity": {
            "n_calls": len(recs),
            "n_json_valid": n_json_valid,
            "n_schema_valid": n_schema_valid,
        },
        "vllm_metrics": {
            "before": before, "after": after,
            "delta": metrics_delta(before, after),
        },
        "plan": res.plan,  # carried for evaluate.py (F1 computation)
    }
    return record


async def run_config(config, manifest, client, model, metrics_url, use_metrics,
                     limit=None):
    entries = manifest[:limit] if limit else manifest
    rows = []
    for entry in entries:
        row = await run_one_patient(client, config, entry, model, metrics_url,
                                    use_metrics)
        rows.append(row)
        wc = row["wall_clock_s"]
        print(f"  [{config}] note_{row['note_id']:02d} "
              f"tokens={row['note_tokens']:5d} latency={wc:6.3f}s "
              f"calls={row['n_llm_calls']} reflect={row['reflection']['n_triggered']}")
    return rows


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _pct(data, q):
    if not data:
        return None
    s = sorted(data)
    if len(s) == 1:
        return s[0]
    idx = q / 100.0 * (len(s) - 1)
    lo = int(idx)
    frac = idx - lo
    if lo + 1 < len(s):
        return s[lo] + frac * (s[lo + 1] - s[lo])
    return s[lo]


def aggregate_config(rows):
    lat = [r["wall_clock_s"] for r in rows]
    ptok = [r["total_prompt_tokens"] for r in rows]
    ctok = [r["total_completion_tokens"] for r in rows]
    n = len(rows)
    total_wall = sum(lat)
    agg = {
        "n_patients": n,
        "latency_s": {
            "mean": stats.mean(lat) if lat else None,
            "median": stats.median(lat) if lat else None,
            "p50": _pct(lat, 50), "p95": _pct(lat, 95), "p99": _pct(lat, 99),
            "min": min(lat) if lat else None, "max": max(lat) if lat else None,
        },
        "throughput_patients_per_min": (n / total_wall * 60) if total_wall else None,
        "prompt_tokens": {"mean": stats.mean(ptok) if ptok else None,
                          "total": sum(ptok)},
        "completion_tokens": {"mean": stats.mean(ctok) if ctok else None,
                              "total": sum(ctok)},
        "reflection_trigger_rate": (
            sum(1 for r in rows if r["reflection"]["n_triggered"] > 0) / n
            if n else None),
        "json_validity_rate": (
            sum(r["validity"]["n_json_valid"] for r in rows) /
            max(1, sum(r["validity"]["n_calls"] for r in rows))),
        "schema_validity_rate": (
            sum(r["validity"]["n_schema_valid"] for r in rows) /
            max(1, sum(r["validity"]["n_calls"] for r in rows))),
    }

    # Per-worker mean latency / TTFT (B/C have 3 workers; A has single_shot).
    workers = {}
    for r in rows:
        for w, d in r["per_worker"].items():
            wl = workers.setdefault(w, {"latency_s": [], "ttft_s": []})
            wl["latency_s"].append(d["latency_s"])
            if d["ttft_s"] is not None:
                wl["ttft_s"].append(d["ttft_s"])
    agg["per_worker"] = {
        w: {"mean_latency_s": stats.mean(v["latency_s"]) if v["latency_s"] else None,
            "mean_ttft_s": stats.mean(v["ttft_s"]) if v["ttft_s"] else None}
        for w, v in workers.items()
    }

    # vLLM-derived: peak GPU cache utilization, mean prefix-cache hit rate.
    hit_rates, cache_use = [], []
    for r in rows:
        delta = r.get("vllm_metrics", {}).get("delta") or {}
        hr = delta.get("vllm:gpu_prefix_cache_hit_rate")
        cu = delta.get("vllm:gpu_cache_usage_perc")
        if hr is not None:
            hit_rates.append(hr)
        if cu is not None:
            cache_use.append(cu)
    agg["prefix_cache_hit_rate_mean"] = stats.mean(hit_rates) if hit_rates else None
    agg["gpu_cache_usage_peak"] = max(cache_use) if cache_use else None
    return agg


def recompute_aggregate(output_dir):
    aggregate = {}
    for config in ["A", "B", "C"]:
        path = os.path.join(output_dir, f"raw_{config}.jsonl")
        if not os.path.exists(path):
            continue
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        aggregate[config] = aggregate_config(rows)
    out = os.path.join(output_dir, "aggregate.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Wrote {out} (configs: {list(aggregate.keys())})")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, choices=["A", "B", "C"])
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--metrics-url", default=None,
                    help="defaults to <base-url root>/metrics")
    ap.add_argument("--model", default=agent.MODEL)
    ap.add_argument("--manifest", default=os.path.join("data", "notes_manifest.json"))
    ap.add_argument("--db", default=os.path.join("data", "synthea.db"))
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mock", action="store_true",
                    help="use the in-process MockClient (no server/GPU)")
    ap.add_argument("--no-metrics", action="store_true",
                    help="skip scraping vLLM /metrics")
    args = ap.parse_args()

    tools.set_db_path(args.db)
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = json.load(open(args.manifest, encoding="utf-8"))

    use_metrics = not args.no_metrics and not args.mock
    metrics_url = args.metrics_url or metrics_url_from_base(args.base_url)

    if args.mock:
        client = agent.MockClient()
        print("Using MockClient (local wiring test).")
    else:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=args.base_url, api_key="dummy")
        print(f"Using AsyncOpenAI at {args.base_url} (metrics: "
              f"{metrics_url if use_metrics else 'disabled'}).")

    print(f"\n=== Config {args.config} over {args.limit or len(manifest)} patients ===")
    rows = asyncio.run(run_config(
        args.config, manifest, client, args.model, metrics_url, use_metrics,
        limit=args.limit))

    raw_path = os.path.join(args.output_dir, f"raw_{args.config}.jsonl")
    with open(raw_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {raw_path} ({len(rows)} patients)")

    recompute_aggregate(args.output_dir)


if __name__ == "__main__":
    main()
