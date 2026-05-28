from __future__ import annotations

import argparse

from hypertrade import optimize_run
from hypertrade.config import DEFAULT_FILTER_SEARCH_SPACE, DEFAULT_OBJECTIVE_PROFILE, OptimizationRunConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", required=True)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--benchmark_name", type=str, default=None)
    parser.add_argument("--run_label", type=str, default=None)
    parser.add_argument("--artifacts_root", type=str, default="artifacts/experiments")
    parser.add_argument("--objective_profile", type=str, default=None)
    args = parser.parse_args()

    run_dir = optimize_run(
        OptimizationRunConfig(
            signals_path=args.signals,
            n_trials=args.n_trials,
            benchmark_name=args.benchmark_name,
            run_label=args.run_label,
            artifacts_root=args.artifacts_root,
            objective_profile_path=args.objective_profile,
        ),
        objective_profile=None if args.objective_profile else DEFAULT_OBJECTIVE_PROFILE,
        search_space=DEFAULT_FILTER_SEARCH_SPACE,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
