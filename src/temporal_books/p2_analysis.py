"""Offline paired-bootstrap analysis for a completed temporal P2 run."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.temporal_books.common import load_yaml, project_path, sha256_file


def paired_bootstrap_ci(values: Sequence[float], *, resamples: int, seed: int) -> tuple[float, float]:
    if not values:
        raise ValueError("paired bootstrap requires at least one value")
    if resamples < 100:
        raise ValueError("bootstrap resamples must be at least 100")
    rng = random.Random(seed)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    means.sort()
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def analyze(metrics: Mapping[str, Any], *, resamples: int, seed: int) -> dict[str, Any]:
    local = metrics["arms"]["local"]["per_event_ndcg_at_5"]
    result: dict[str, Any] = {
        "n_events": len(local),
        "metric": "paired NDCG@5 difference versus local",
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "arms": {},
    }
    for arm in ("one_hop", "oracle_two_hop"):
        values = metrics["arms"][arm]["per_event_ndcg_at_5"]
        delta = [float(value) - float(reference) for value, reference in zip(values, local)]
        lower, upper = paired_bootstrap_ci(delta, resamples=resamples, seed=seed)
        result["arms"][arm] = {
            "mean_delta_ndcg_at_5_vs_local": sum(delta) / len(delta),
            "bootstrap_95_ci": [lower, upper],
            "improved_events": sum(value > 0 for value in delta),
            "worsened_events": sum(value < 0 for value in delta),
            "unchanged_events": sum(value == 0 for value in delta),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline P2 paired-bootstrap analysis")
    parser.add_argument("--config", default="configs/temporal_amazon_books_2014/p2_v2_smoke.yaml")
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    p2 = config["p2"]
    metrics_path = project_path(p2["metrics"])
    output_path = project_path(p2.get("analysis", metrics_path.with_name(metrics_path.name.replace("_metrics.json", "_analysis.json"))))
    if not metrics_path.exists():
        raise FileNotFoundError(f"P2 metrics missing: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    analysis = analyze(
        metrics,
        resamples=int(p2.get("bootstrap_resamples", 10_000)),
        seed=int(p2.get("bootstrap_seed", 20260826)),
    )
    analysis.update(
        {
            "run_id": p2["run_id"],
            "metrics": {"path": str(metrics_path.relative_to(project_path("."))), "sha256": sha256_file(metrics_path)},
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
