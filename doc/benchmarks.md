# Benchmark smoke

```sh
CHECK_MODE=benchmark bash doc/ai/project-checks.sh
```

Runs existing checks, then one seeded NSGA-II trial plus baseline on each unchanged `benchmarks/fixtures/PF20250597.csv` and `benchmarks/fixtures/signals.csv`. No input CSV, source OHLC or source indicator is rewritten. Local market-data coverage determines eligibility; missing files are visible rejections.

Each study must persist a full schema-3 bundle. The suite links study paths and records success/no_valid_candidates/failed and artifact completeness. A failed or incomplete study causes a nonzero CLI exit. A correct study with no candidates remains a successful smoke outcome, rather than a forced profitable result.

The first fixture uses ISO dates; the loader additionally accepts dd.mm.yyyy and preserves explicit UTC instants. The second historical fixture's `shotr_symbols` spelling is supported read-only. Numbers from the former execution/statistics schema are superseded by T-194 and should not be compared directly.
