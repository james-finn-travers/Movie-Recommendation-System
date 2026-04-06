from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run benchmark evaluation and generate markdown report in one command.",
    )

    # Benchmark args
    p.add_argument("--data-dir", type=Path, default=ROOT / "ml-latest")
    p.add_argument("--movie-tags", type=Path, default=ROOT / "movie_tags.npz")
    p.add_argument("--prefs", type=Path, default=ROOT / "users_preferences.dat")
    p.add_argument("--user-index", type=Path, default=ROOT / "user_index_to_id.npy")
    p.add_argument("--sample-users", type=int, default=600)
    p.add_argument("--benchmark-step", type=int, default=150)
    p.add_argument("--min-ratings", type=int, default=20)
    p.add_argument("--min-pos-rating", type=float, default=3.5)
    p.add_argument("--k-neighbors", type=int, default=25)
    p.add_argument("--min-neighbor-votes", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-path", type=Path, default=ROOT / "benchmark_results.csv")

    # Report args
    p.add_argument("--report-path", type=Path, default=ROOT / "benchmark_report.md")
    p.add_argument("--max-p95-ms", type=float, default=150.0)
    p.add_argument("--max-p99-ms", type=float, default=250.0)
    p.add_argument("--max-recall10-drop-pct", type=float, default=5.0)
    p.add_argument("--max-ndcg10-drop-pct", type=float, default=5.0)

    # Runtime
    p.add_argument(
        "--python-exec",
        default=sys.executable,
        help="Python executable used to run child scripts.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    eval_cmd = [
        args.python_exec,
        str(ROOT / "evaluate_model.py"),
        "--data-dir",
        str(args.data_dir),
        "--movie-tags",
        str(args.movie_tags),
        "--prefs",
        str(args.prefs),
        "--user-index",
        str(args.user_index),
        "--sample-users",
        str(args.sample_users),
        "--benchmark-step",
        str(args.benchmark_step),
        "--min-ratings",
        str(args.min_ratings),
        "--min-pos-rating",
        str(args.min_pos_rating),
        "--k-neighbors",
        str(args.k_neighbors),
        "--min-neighbor-votes",
        str(args.min_neighbor_votes),
        "--seed",
        str(args.seed),
        "--results-path",
        str(args.results_path),
    ]

    report_cmd = [
        args.python_exec,
        str(ROOT / "benchmark_report.py"),
        "--results-path",
        str(args.results_path),
        "--output-md",
        str(args.report_path),
        "--max-p95-ms",
        str(args.max_p95_ms),
        "--max-p99-ms",
        str(args.max_p99_ms),
        "--max-recall10-drop-pct",
        str(args.max_recall10_drop_pct),
        "--max-ndcg10-drop-pct",
        str(args.max_ndcg10_drop_pct),
    ]

    print("Running benchmark evaluator...")
    subprocess.run(eval_cmd, check=True)

    print("Generating benchmark report...")
    subprocess.run(report_cmd, check=True)

    print(f"Done. Results: {args.results_path}")
    print(f"Done. Report: {args.report_path}")


if __name__ == "__main__":
    main()
