from __future__ import annotations

import argparse
import json

from hypertrade import run_benchmark_suite
from hypertrade.config.schemas import FilterSearchSpace, ObjectiveProfile
from hypertrade.experiments.benchmarks import BENCHMARKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS.keys()))
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--artifacts_root", type=str, default="artifacts/experiments")
    parser.add_argument("--benchmark_root", type=str, default="artifacts/benchmarks")
    parser.add_argument("--objective_profile", type=str, default=None)
    parser.add_argument("--search_space", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    objective_profile = ObjectiveProfile.load(args.objective_profile) if args.objective_profile else None
    search_space = FilterSearchSpace.load(args.search_space) if args.search_space else None
    suite_dir = run_benchmark_suite(
        benchmark_names=args.benchmarks,
        n_trials=args.n_trials,
        artifacts_root=args.artifacts_root,
        benchmark_root=args.benchmark_root,
        objective_profile=objective_profile,
        objective_profile_path=args.objective_profile,
        search_space=search_space,
        seed=args.seed,
    )
    print(suite_dir)
    summary = json.loads((suite_dir/'summary.json').read_text())
    if summary['failure_count'] or summary['artifact_complete_count'] != len(summary['benchmark_names']):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
