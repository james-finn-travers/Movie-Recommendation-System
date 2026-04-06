from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NUMERIC_FIELDS = {
    "seed": int,
    "users_evaluated": int,
    "benchmark_step": int,
    "k_neighbors": int,
    "min_neighbor_votes": int,
    "precision_at_5": float,
    "recall_at_5": float,
    "ndcg_at_5": float,
    "precision_at_10": float,
    "recall_at_10": float,
    "ndcg_at_10": float,
    "precision_at_20": float,
    "recall_at_20": float,
    "ndcg_at_20": float,
    "latency_p50_ms": float,
    "latency_p95_ms": float,
    "latency_p99_ms": float,
    "total_runtime_sec": float,
}


@dataclass
class GateResult:
    name: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate benchmark summary and regression gates from benchmark_results.csv",
    )
    p.add_argument("--results-path", type=Path, default=Path("./benchmark_results.csv"))
    p.add_argument("--output-md", type=Path, default=Path("./benchmark_report.md"))
    p.add_argument("--max-p95-ms", type=float, default=150.0)
    p.add_argument("--max-p99-ms", type=float, default=250.0)
    p.add_argument("--max-recall10-drop-pct", type=float, default=5.0)
    p.add_argument("--max-ndcg10-drop-pct", type=float, default=5.0)
    return p.parse_args()


def parse_row(raw_row: dict[str, str]) -> dict[str, Any]:
    row: dict[str, Any] = dict(raw_row)
    for key, caster in NUMERIC_FIELDS.items():
        if key in row and row[key] != "":
            row[key] = caster(row[key])
    return row


def load_run_final_rows(results_path: Path) -> list[dict[str, Any]]:
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with results_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = parse_row(raw_row)
            grouped[row["run_timestamp"]].append(row)

    final_rows: list[dict[str, Any]] = []
    for run_ts, rows in grouped.items():
        _ = run_ts
        final_rows.append(max(rows, key=lambda r: int(r["users_evaluated"])))

    final_rows.sort(key=lambda r: str(r["run_timestamp"]))
    return final_rows


def pct_delta(curr: float, prev: float) -> float:
    if prev == 0:
        if curr == 0:
            return 0.0
        return float("inf")
    return ((curr - prev) / prev) * 100.0


def fmt_delta(curr: float, prev: float) -> str:
    d = pct_delta(curr, prev)
    if d == float("inf"):
        return "+inf%"
    return f"{d:+.2f}%"


def evaluate_gates(
    latest: dict[str, Any],
    previous: dict[str, Any] | None,
    max_p95_ms: float,
    max_p99_ms: float,
    max_recall10_drop_pct: float,
    max_ndcg10_drop_pct: float,
) -> list[GateResult]:
    gates: list[GateResult] = []

    p95 = float(latest["latency_p95_ms"])
    p99 = float(latest["latency_p99_ms"])
    gates.append(
        GateResult(
            name="Latency p95",
            status="PASS" if p95 <= max_p95_ms else "FAIL",
            detail=f"{p95:.2f} ms <= {max_p95_ms:.2f} ms",
        )
    )
    gates.append(
        GateResult(
            name="Latency p99",
            status="PASS" if p99 <= max_p99_ms else "FAIL",
            detail=f"{p99:.2f} ms <= {max_p99_ms:.2f} ms",
        )
    )

    if previous is None:
        gates.append(
            GateResult(
                name="Recall@10 regression",
                status="PASS",
                detail="No previous run to compare.",
            )
        )
        gates.append(
            GateResult(
                name="NDCG@10 regression",
                status="PASS",
                detail="No previous run to compare.",
            )
        )
        return gates

    recall_drop = -pct_delta(float(latest["recall_at_10"]), float(previous["recall_at_10"]))
    ndcg_drop = -pct_delta(float(latest["ndcg_at_10"]), float(previous["ndcg_at_10"]))

    gates.append(
        GateResult(
            name="Recall@10 regression",
            status="PASS" if recall_drop <= max_recall10_drop_pct else "FAIL",
            detail=(
                f"drop={max(0.0, recall_drop):.2f}% <= {max_recall10_drop_pct:.2f}%"
            ),
        )
    )
    gates.append(
        GateResult(
            name="NDCG@10 regression",
            status="PASS" if ndcg_drop <= max_ndcg10_drop_pct else "FAIL",
            detail=(
                f"drop={max(0.0, ndcg_drop):.2f}% <= {max_ndcg10_drop_pct:.2f}%"
            ),
        )
    )
    return gates


def render_markdown(
    latest: dict[str, Any],
    previous: dict[str, Any] | None,
    gates: list[GateResult],
) -> str:
    lines: list[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"Latest run timestamp: {latest['run_timestamp']}")
    lines.append(f"Users evaluated: {latest['users_evaluated']}")
    lines.append(f"Seed: {latest['seed']}")
    lines.append("")

    lines.append("## Latest Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Precision@10 | {float(latest['precision_at_10']):.4f} |")
    lines.append(f"| Recall@10 | {float(latest['recall_at_10']):.4f} |")
    lines.append(f"| NDCG@10 | {float(latest['ndcg_at_10']):.4f} |")
    lines.append(f"| Latency p50 (ms) | {float(latest['latency_p50_ms']):.2f} |")
    lines.append(f"| Latency p95 (ms) | {float(latest['latency_p95_ms']):.2f} |")
    lines.append(f"| Latency p99 (ms) | {float(latest['latency_p99_ms']):.2f} |")
    lines.append(f"| Total runtime (s) | {float(latest['total_runtime_sec']):.2f} |")
    lines.append("")

    lines.append("## Delta vs Previous Run")
    lines.append("")
    if previous is None:
        lines.append("No previous run found. Delta is unavailable.")
        lines.append("")
    else:
        lines.append("| Metric | Previous | Latest | Delta |")
        lines.append("|---|---:|---:|---:|")
        delta_fields = [
            "precision_at_10",
            "recall_at_10",
            "ndcg_at_10",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
        ]
        for field in delta_fields:
            prev_val = float(previous[field])
            latest_val = float(latest[field])
            lines.append(
                f"| {field} | {prev_val:.4f} | {latest_val:.4f} | {fmt_delta(latest_val, prev_val)} |"
            )
        lines.append("")

    lines.append("## Gate Checks")
    lines.append("")
    lines.append("| Gate | Status | Details |")
    lines.append("|---|---|---|")
    for gate in gates:
        lines.append(f"| {gate.name} | {gate.status} | {gate.detail} |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    final_rows = load_run_final_rows(args.results_path)
    if not final_rows:
        raise ValueError("No benchmark rows found in results file.")

    latest = final_rows[-1]
    previous = final_rows[-2] if len(final_rows) >= 2 else None

    gates = evaluate_gates(
        latest=latest,
        previous=previous,
        max_p95_ms=args.max_p95_ms,
        max_p99_ms=args.max_p99_ms,
        max_recall10_drop_pct=args.max_recall10_drop_pct,
        max_ndcg10_drop_pct=args.max_ndcg10_drop_pct,
    )

    markdown = render_markdown(latest=latest, previous=previous, gates=gates)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown + "\n", encoding="utf-8")

    print(f"Wrote report: {args.output_md}")
    print(f"Latest run: {latest['run_timestamp']} | users={latest['users_evaluated']}")
    print("Gate summary:")
    for gate in gates:
        print(f"  - {gate.name}: {gate.status} ({gate.detail})")


if __name__ == "__main__":
    main()
