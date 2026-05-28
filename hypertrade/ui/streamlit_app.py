from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hypertrade.ui.analysis import (
    render_benchmarks,
    render_experiments,
    render_filters,
    render_overview,
    render_pareto,
    render_trades,
    render_trials,
)
from hypertrade.ui.home import render_home
from hypertrade.ui.launcher import render_launcher
from hypertrade.ui.objectives import render_objectives
from hypertrade.ui.state import run_options


def main() -> None:
    st.set_page_config(page_title="HyperTrade", layout="wide")
    st.title("HyperTrade Research Dashboard")
    st.caption("Use `run_dashboard` for the UI. `run_benchmarks` runs studies in the terminal and exits.")

    runs_df = run_options()
    page = st.sidebar.radio(
        "Page",
        [
            "Home",
            "Overview",
            "Experiments",
            "Launcher",
            "Objectives",
            "Pareto",
            "Trials",
            "Trades",
            "Filters",
            "Benchmarks",
        ],
    )

    selected_run = None
    if not runs_df.empty:
        selected_run = st.sidebar.selectbox("Run", runs_df["path"].tolist())

    if page == "Home":
        render_home()
    elif page == "Overview":
        render_overview(runs_df)
    elif page == "Experiments":
        render_experiments(runs_df)
    elif page == "Launcher":
        render_launcher()
    elif page == "Objectives":
        render_objectives(selected_run)
    elif page == "Pareto":
        render_pareto(selected_run)
    elif page == "Trials":
        render_trials(selected_run)
    elif page == "Trades":
        render_trades(selected_run)
    elif page == "Filters":
        render_filters(selected_run)
    elif page == "Benchmarks":
        render_benchmarks(runs_df)


if __name__ == "__main__":
    main()
