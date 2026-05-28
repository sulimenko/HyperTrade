from __future__ import annotations

import streamlit as st


def render_home() -> None:
    st.header("Home")
    st.markdown(
        """
HyperTrade has two different runtime modes:

- `run_dashboard`: starts the Streamlit research UI
- `run_benchmarks`: runs benchmark studies in the terminal and then exits

If the browser is open and you run `run_benchmarks` instead of `run_dashboard`, the old Streamlit session will disconnect and the browser will show a connection error. In that case, just start the dashboard again.
"""
    )

    st.subheader("Quick Start")
    st.code("python run_dashboard.py", language="bash")
    st.markdown(
        """
Then open:

- `http://localhost:8501`

Use the left sidebar for navigation.
"""
    )

    st.subheader("Recommended Flow")
    st.markdown(
        """
1. Open `Objectives` and create or load an objective profile.
2. Open `Launcher` and choose a benchmark or custom signals file.
3. Set `Run label`, `Trials`, and the search space.
4. Run optimization.
5. Review results in `Pareto`, `Trials`, `Trades`, and `Filters`.
6. Use `Benchmarks` to compare benchmark suites and runs.
"""
    )

    st.subheader("What Each Page Does")
    st.markdown(
        """
- `Overview`: recent runs, top candidates, benchmark suite status
- `Experiments`: filterable run list
- `Launcher`: start a new optimization run
- `Objectives`: configure objectives, constraints, and candidate policy
- `Pareto`: frontier analysis and candidate ranking
- `Trials`: optimization trace and parameter interactions
- `Trades`: trade-level behavior and outcomes
- `Filters`: signal acceptance and conversion analysis
- `Benchmarks`: benchmark suite and run comparison
"""
    )
