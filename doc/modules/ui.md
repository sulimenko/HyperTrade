# UI Module

Code location: `hypertrade/ui`.

The UI module contains the Streamlit dashboard. `streamlit_app.py` wires the page navigation, while page modules render overview, experiments, launcher, objectives, Pareto, trials, trades, filters, and benchmark analysis.

`state.py` centralizes artifact paths, default objective/search-space initialization, run listing, manifest loading, and safe table reads used by the pages.

Supported launch command:

```bash
python run_dashboard.py
```

Important boundaries:

- The dashboard is for research workflow and analysis.
- `run_dashboard.py` starts a long-running UI process.
- `run_benchmarks.py` is a terminal benchmark runner and exits when complete.
