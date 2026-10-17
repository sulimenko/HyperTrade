from __future__ import annotations

import argparse
import json

from hypertrade import optimize_run
from hypertrade.config.defaults import DEFAULT_FILTER_SEARCH_SPACE, DEFAULT_OBJECTIVE_PROFILE
from hypertrade.config.schemas import FilterSearchSpace, OptimizationRunConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", required=True)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--benchmark_name", type=str, default=None)
    parser.add_argument("--run_label", type=str, default=None)
    parser.add_argument("--artifacts_root", type=str, default="artifacts/experiments")
    parser.add_argument("--objective_profile", type=str, default=None)
    parser.add_argument("--search_space", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = optimize_run(
        OptimizationRunConfig(
            signals_path=args.signals,
            n_trials=args.n_trials,
            benchmark_name=args.benchmark_name,
            run_label=args.run_label,
            artifacts_root=args.artifacts_root,
            objective_profile_path=args.objective_profile,
            seed=args.seed,
        ),
        objective_profile=None if args.objective_profile else DEFAULT_OBJECTIVE_PROFILE,
        search_space=FilterSearchSpace.load(args.search_space) if args.search_space else DEFAULT_FILTER_SEARCH_SPACE,
    )
    print(run_dir)
    manifest = run_dir/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()).get('status') == 'failed':
        raise SystemExit(1)


if __name__ == "__main__":
    main()
