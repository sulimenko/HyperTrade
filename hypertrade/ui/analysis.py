from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hypertrade.ui.state import (
    benchmark_suite_options,
    load_manifest,
    load_metrics_summary,
    load_run_table_safe,
    numeric_columns,
    safe_metric,
)


def render_overview(runs_df) -> None:
    st.header("Overview")
    if runs_df.empty:
        st.info("No experiment runs found.")
        return

    total_runs = len(runs_df)
    benchmark_count = runs_df["benchmark_name"].nunique(dropna=True)
    latest_run = runs_df.iloc[0]["run_id"]
    suite_df = benchmark_suite_options()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Runs", total_runs)
    col2.metric("Benchmarks", benchmark_count)
    col3.metric("Latest run", latest_run)
    col4.metric("Benchmark suites", len(suite_df))
    st.subheader("Recent Runs")
    st.dataframe(runs_df.head(10), width="stretch")

    benchmark_counts = runs_df.groupby("benchmark_name", dropna=False).size().reset_index(name="run_count")
    st.plotly_chart(px.bar(benchmark_counts, x="benchmark_name", y="run_count", title="Runs by benchmark"), width="stretch")

    if {"created_at", "benchmark_name"}.issubset(runs_df.columns):
        trend_df = runs_df.copy()
        trend_df["created_at"] = pd.to_datetime(trend_df["created_at"], errors="coerce")
        trend_df = trend_df.dropna(subset=["created_at"]).sort_values("created_at")
        if not trend_df.empty:
            trend_df["run_count"] = 1
            trend_df["run_date"] = trend_df["created_at"].dt.date
            daily = trend_df.groupby(["run_date", "benchmark_name"], dropna=False)["run_count"].sum().reset_index()
            st.plotly_chart(px.line(daily, x="run_date", y="run_count", color="benchmark_name", title="Run activity over time"), width="stretch")

    top_rows = []
    for run_path in runs_df["path"].tolist()[:10]:
        shortlist_df = load_run_table_safe(run_path, "candidate_shortlist.parquet")
        if shortlist_df.empty:
            continue
        row = shortlist_df.iloc[0].to_dict()
        row["run_path"] = run_path
        top_rows.append(row)
    top_df = pd.DataFrame(top_rows)
    if not top_df.empty:
        st.subheader("Top Pareto Candidates From Recent Runs")
        display_cols = [col for col in ["run_path", "benchmark_name", "number", "total_pnl", "max_drawdown", "profit_factor", "avg_hold_minutes", "utopia_distance"] if col in top_df.columns]
        st.dataframe(top_df[display_cols], width="stretch")

    if not suite_df.empty:
        st.subheader("Benchmark Suite Status")
        st.dataframe(suite_df.head(10), width="stretch")


def render_experiments(runs_df) -> None:
    st.header("Experiments")
    if runs_df.empty:
        st.info("No experiment runs found.")
        return

    filtered = runs_df.copy()
    options = sorted(filtered["benchmark_name"].dropna().unique().tolist())
    benchmark_filter = st.multiselect("Benchmark filter", options, default=options)
    profile_options = sorted(filtered["objective_profile_name"].dropna().astype(str).unique().tolist())
    profile_filter = st.multiselect("Objective profile", profile_options, default=profile_options)
    max_trials = int(pd.to_numeric(filtered["n_trials"], errors="coerce").fillna(0).max() or 0)
    trial_range = st.slider("Study size", min_value=0, max_value=max(max_trials, 1), value=(0, max(max_trials, 1)))
    created_at = pd.to_datetime(filtered["created_at"], errors="coerce")
    date_values = created_at.dropna()
    date_range = None
    if not date_values.empty:
        date_range = st.date_input("Created between", value=(date_values.min().date(), date_values.max().date()))

    filtered = runs_df.copy()
    if benchmark_filter:
        filtered = filtered[filtered["benchmark_name"].isin(benchmark_filter)]
    if profile_filter:
        filtered = filtered[filtered["objective_profile_name"].astype(str).isin(profile_filter)]
    filtered["n_trials_num"] = pd.to_numeric(filtered["n_trials"], errors="coerce").fillna(0)
    filtered = filtered[(filtered["n_trials_num"] >= trial_range[0]) & (filtered["n_trials_num"] <= trial_range[1])]
    filtered["created_at"] = pd.to_datetime(filtered["created_at"], errors="coerce")
    if date_range and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[(filtered["created_at"].dt.date >= start) & (filtered["created_at"].dt.date <= end)]

    enrich = []
    for run_path in filtered["path"].tolist()[:50]:
        metrics = load_metrics_summary(run_path)
        enrich.append(
            {
                "path": run_path,
                "best_shortlist_total_pnl": metrics.get("best_shortlist_total_pnl"),
                "best_shortlist_profit_factor": metrics.get("best_shortlist_profit_factor"),
                "best_shortlist_min_drawdown": metrics.get("best_shortlist_min_drawdown"),
                "pareto_candidate_count": metrics.get("pareto_candidate_count"),
            }
        )
    if enrich:
        filtered = filtered.merge(pd.DataFrame(enrich), on="path", how="left")
    st.dataframe(filtered, width="stretch")


def render_pareto(selected_run: str | None) -> None:
    st.header("Pareto")
    if not selected_run:
        st.info("Select a run.")
        return

    pareto_df = load_run_table_safe(selected_run, "pareto_trials.parquet")
    if pareto_df.empty:
        st.info("No Pareto trials found.")
        return

    numeric_cols = numeric_columns(pareto_df, excluded={"number", "constraint_violations", "signals_path", "benchmark_name"})
    st.dataframe(pareto_df, width="stretch")
    if len(numeric_cols) >= 2:
        x_col = st.selectbox("X axis", numeric_cols, index=0)
        y_col = st.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1))
        color_col = st.selectbox("Color", ["<none>"] + numeric_cols, index=min(2, len(numeric_cols)))
        st.plotly_chart(px.scatter(pareto_df, x=x_col, y=y_col, color=None if color_col == "<none>" else color_col, hover_data=["number"], title="Pareto Candidates"), width="stretch")
        if len(numeric_cols) >= 3:
            z_col = st.selectbox("3D axis", numeric_cols, index=min(2, len(numeric_cols) - 1))
            st.plotly_chart(px.scatter_3d(pareto_df, x=x_col, y=y_col, z=z_col, color=None if color_col == "<none>" else color_col, hover_data=["number"], title="Pareto Candidates 3D"), width="stretch")
        interaction_cols = [col for col in numeric_cols if col != "number"]
        if len(interaction_cols) >= 3:
            matrix_cols = st.multiselect("Scatter matrix columns", interaction_cols, default=interaction_cols[: min(4, len(interaction_cols))])
            if len(matrix_cols) >= 2:
                st.plotly_chart(px.scatter_matrix(pareto_df, dimensions=matrix_cols, color=None if color_col == "<none>" else color_col, hover_data=["number"], title="Parameter and metric interaction matrix"), width="stretch")
        corr_candidates = [col for col in numeric_cols if pareto_df[col].nunique(dropna=True) > 1]
        if len(corr_candidates) >= 2:
            corr = pareto_df[corr_candidates].corr(numeric_only=True)
            heatmap = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(), colorscale="RdBu", zmid=0.0))
            heatmap.update_layout(title="Pareto correlation heatmap")
            st.plotly_chart(heatmap, width="stretch")

    if "utopia_distance" in pareto_df.columns:
        ranked = pareto_df.sort_values("utopia_distance").head(15)
        st.subheader("Closest to Utopia")
        display_cols = [col for col in ["number", "utopia_distance", "total_pnl", "max_drawdown", "profit_factor", "avg_hold_minutes"] if col in ranked.columns]
        st.dataframe(ranked[display_cols], width="stretch")


def render_trials(selected_run: str | None) -> None:
    st.header("Trials")
    if not selected_run:
        st.info("Select a run.")
        return

    trials_df = load_run_table_safe(selected_run, "trials.parquet")
    if trials_df.empty:
        st.info("No trials found.")
        return

    numeric_cols = numeric_columns(trials_df, excluded={"number"})
    if {"number"}.issubset(trials_df.columns) and numeric_cols:
        metric = st.selectbox("Trial metric", numeric_cols, index=0)
        st.plotly_chart(px.line(trials_df.sort_values("number"), x="number", y=metric, title=f"{metric} by trial"), width="stretch")
        rolling_window = min(10, max(len(trials_df), 1))
        trend_df = trials_df.sort_values("number").copy()
        trend_df[f"{metric}_rolling_mean"] = pd.to_numeric(trend_df[metric], errors="coerce").rolling(rolling_window, min_periods=1).mean()
        st.plotly_chart(px.line(trend_df, x="number", y=[metric, f"{metric}_rolling_mean"], title=f"{metric} and rolling mean"), width="stretch")

    param_cols = [col for col in trials_df.columns if col.startswith("params_")]
    metric_cols = [col for col in numeric_cols if not col.startswith("values_")]
    if param_cols and metric_cols:
        color_metric = st.selectbox("Parameter interaction color", metric_cols, index=0)
        x_param = st.selectbox("X parameter", param_cols, index=0)
        y_param = st.selectbox("Y parameter", param_cols, index=min(1, len(param_cols) - 1))
        st.plotly_chart(px.scatter(trials_df, x=x_param, y=y_param, color=color_metric, hover_data=["number"], title="Trial parameter interaction"), width="stretch")
        corr_cols = [col for col in param_cols + metric_cols if pd.api.types.is_numeric_dtype(trials_df[col]) and trials_df[col].nunique(dropna=True) > 1]
        if len(corr_cols) >= 2:
            corr = trials_df[corr_cols].corr(numeric_only=True)
            heatmap = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(), colorscale="Viridis"))
            heatmap.update_layout(title="Trial parameter/metric correlation heatmap")
            st.plotly_chart(heatmap, width="stretch")
    st.dataframe(trials_df, width="stretch")


def render_trades(selected_run: str | None) -> None:
    st.header("Trades")
    if not selected_run:
        st.info("Select a run.")
        return

    trades_df = load_run_table_safe(selected_run, "trades.parquet")
    if trades_df.empty:
        st.info("No trades found.")
        return

    trades_df = trades_df.copy()
    if "direction" in trades_df.columns:
        trades_df["side"] = trades_df["direction"].map({1: "long", -1: "short"}).fillna("unknown")
    if "entry_dt" in trades_df.columns:
        trades_df["entry_dt"] = pd.to_datetime(trades_df["entry_dt"], errors="coerce")
        trades_df["entry_weekday"] = trades_df["entry_dt"].dt.day_name()
        trades_df["entry_hour"] = trades_df["entry_dt"].dt.tz_convert("America/New_York").dt.hour if trades_df["entry_dt"].dt.tz is not None else trades_df["entry_dt"].dt.hour
    if "exit_dt" in trades_df.columns:
        trades_df["exit_dt"] = pd.to_datetime(trades_df["exit_dt"], errors="coerce")

    st.dataframe(trades_df.head(200), width="stretch")
    if {"exit_dt", "pnl"}.issubset(trades_df.columns):
        chart_df = trades_df.copy().sort_values("exit_dt")
        chart_df["equity"] = chart_df["pnl"].cumsum()
        chart_df["peak"] = chart_df["equity"].cummax()
        chart_df["drawdown"] = chart_df["equity"] - chart_df["peak"]
        st.plotly_chart(px.line(chart_df, x="exit_dt", y="equity", color="candidate_trial", title="Equity Curve"), width="stretch")
        st.plotly_chart(px.line(chart_df, x="exit_dt", y="drawdown", color="candidate_trial", title="Drawdown Curve"), width="stretch")
        if "is_win" in chart_df.columns:
            chart_df["rolling_win_rate"] = chart_df["is_win"].astype(float).rolling(25, min_periods=5).mean()
            st.plotly_chart(px.line(chart_df, x="exit_dt", y="rolling_win_rate", color="candidate_trial", title="Rolling Win Rate"), width="stretch")
    if "return_pct" in trades_df.columns:
        st.plotly_chart(px.histogram(trades_df, x="return_pct", color="candidate_trial", nbins=50, title="Trade Return Distribution"), width="stretch")
    if "hold_minutes" in trades_df.columns:
        st.plotly_chart(px.histogram(trades_df, x="hold_minutes", color="candidate_trial", nbins=40, title="Hold Time Distribution"), width="stretch")
    if "exit_reason" in trades_df.columns:
        exit_mix = trades_df.groupby(["candidate_trial", "exit_reason"]).size().reset_index(name="count")
        st.plotly_chart(px.bar(exit_mix, x="candidate_trial", y="count", color="exit_reason", title="Exit Reason Mix"), width="stretch")
    if "symbol" in trades_df.columns:
        symbol_mix = trades_df.groupby("symbol", dropna=False)["pnl"].sum().sort_values(ascending=False).head(20).reset_index()
        st.plotly_chart(px.bar(symbol_mix, x="symbol", y="pnl", title="Top Symbol Contribution"), width="stretch")
    if {"side", "pnl"}.issubset(trades_df.columns):
        side_summary = trades_df.groupby(["candidate_trial", "side"], dropna=False).agg(pnl=("pnl", "sum"), trades=("symbol", "count"), avg_return_pct=("return_pct", "mean"), win_rate=("is_win", "mean")).reset_index()
        st.subheader("Long/Short Split")
        st.dataframe(side_summary, width="stretch")
        st.plotly_chart(px.bar(side_summary, x="candidate_trial", y="pnl", color="side", barmode="group", title="PnL by side"), width="stretch")
    if {"entry_weekday", "pnl"}.issubset(trades_df.columns):
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_df = trades_df.groupby("entry_weekday", dropna=False)["pnl"].sum().reset_index()
        weekday_df["entry_weekday"] = pd.Categorical(weekday_df["entry_weekday"], categories=weekday_order, ordered=True)
        weekday_df = weekday_df.sort_values("entry_weekday")
        st.plotly_chart(px.bar(weekday_df, x="entry_weekday", y="pnl", title="PnL by entry weekday"), width="stretch")
    if {"entry_hour", "pnl"}.issubset(trades_df.columns):
        hour_df = trades_df.groupby("entry_hour", dropna=False)["pnl"].agg(["sum", "count"]).reset_index()
        hour_df.columns = ["entry_hour", "pnl", "trade_count"]
        st.plotly_chart(px.bar(hour_df, x="entry_hour", y="pnl", title="PnL by entry hour (NY time)"), width="stretch")


def render_filters(selected_run: str | None) -> None:
    st.header("Filters")
    if not selected_run:
        st.info("Select a run.")
        return

    shortlist_df = load_run_table_safe(selected_run, "candidate_shortlist.parquet")
    signal_stats_df = load_run_table_safe(selected_run, "signal_stats.parquet")
    if shortlist_df.empty:
        st.info("No shortlisted candidates found.")
        return
    st.dataframe(shortlist_df, width="stretch")
    if not signal_stats_df.empty:
        signal_stats_df = signal_stats_df.copy()
        if "datetime" in signal_stats_df.columns:
            signal_stats_df["datetime"] = pd.to_datetime(signal_stats_df["datetime"], errors="coerce")
            signal_stats_df["weekday"] = signal_stats_df["datetime"].dt.day_name()
            signal_stats_df["hour"] = signal_stats_df["datetime"].dt.tz_convert("America/New_York").dt.hour if signal_stats_df["datetime"].dt.tz is not None else signal_stats_df["datetime"].dt.hour
        if {"candidate_trial", "symbols_total", "symbols_traded", "symbols_rejected"}.issubset(signal_stats_df.columns):
            funnel_df = signal_stats_df.groupby("candidate_trial")[["symbols_total", "symbols_traded", "symbols_rejected"]].sum().reset_index()
            melt = funnel_df.melt(id_vars="candidate_trial", var_name="stage", value_name="count")
            st.plotly_chart(px.bar(melt, x="candidate_trial", y="count", color="stage", title="Signal Conversion Funnel"), width="stretch")
        if {"datetime", "total_pnl", "candidate_trial"}.issubset(signal_stats_df.columns):
            pnl_df = signal_stats_df.sort_values("datetime").copy()
            pnl_df["cum_signal_pnl"] = pnl_df.groupby("candidate_trial")["total_pnl"].cumsum()
            st.plotly_chart(px.line(pnl_df, x="datetime", y="total_pnl", color="candidate_trial", title="Signal-level PnL"), width="stretch")
            st.plotly_chart(px.line(pnl_df, x="datetime", y="cum_signal_pnl", color="candidate_trial", title="Cumulative Signal-level PnL"), width="stretch")
        if {"accepted_long", "accepted_short", "input_long", "input_short", "candidate_trial"}.issubset(signal_stats_df.columns):
            side_df = signal_stats_df.groupby("candidate_trial")[["input_long", "input_short", "accepted_long", "accepted_short"]].sum().reset_index()
            side_melt = side_df.melt(id_vars="candidate_trial", var_name="stage", value_name="count")
            st.plotly_chart(px.bar(side_melt, x="candidate_trial", y="count", color="stage", barmode="group", title="Acceptance by side"), width="stretch")
        if {"weekday", "total_pnl"}.issubset(signal_stats_df.columns):
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday_df = signal_stats_df.groupby("weekday", dropna=False)["total_pnl"].sum().reset_index()
            weekday_df["weekday"] = pd.Categorical(weekday_df["weekday"], categories=weekday_order, ordered=True)
            weekday_df = weekday_df.sort_values("weekday")
            st.plotly_chart(px.bar(weekday_df, x="weekday", y="total_pnl", title="Signal PnL by weekday"), width="stretch")
        if {"hour", "total_pnl"}.issubset(signal_stats_df.columns):
            hour_df = signal_stats_df.groupby("hour", dropna=False)["total_pnl"].sum().reset_index()
            st.plotly_chart(px.bar(hour_df, x="hour", y="total_pnl", title="Signal PnL by hour (NY time)"), width="stretch")


def render_benchmarks(runs_df) -> None:
    st.header("Benchmarks")
    suite_df = benchmark_suite_options()
    if not suite_df.empty:
        st.subheader("Benchmark Suites")
        st.dataframe(suite_df, width="stretch")
        selected_suite = st.selectbox("Inspect suite", suite_df["path"].tolist())
        suite_runs = load_run_table_safe(selected_suite, "benchmark_runs.parquet")
        if not suite_runs.empty:
            st.dataframe(suite_runs, width="stretch")
            st.plotly_chart(px.bar(suite_runs, x="benchmark_name", y="duration_seconds", color="status", title="Suite duration by benchmark"), width="stretch")
            if {"benchmark_name", "best_shortlist_total_pnl"}.issubset(suite_runs.columns):
                st.plotly_chart(px.bar(suite_runs, x="benchmark_name", y="best_shortlist_total_pnl", color="status", title="Best shortlist total pnl by benchmark"), width="stretch")

    if runs_df.empty:
        if suite_df.empty:
            st.info("No benchmark runs found.")
        return

    summary = runs_df.groupby("benchmark_name", dropna=False).size().reset_index(name="run_count")
    st.dataframe(summary, width="stretch")
    st.plotly_chart(px.bar(summary, x="benchmark_name", y="run_count", title="Benchmark Run Counts"), width="stretch")

    compare_runs = st.multiselect("Compare runs", runs_df["path"].tolist(), default=runs_df["path"].tolist()[:2])
    compare_rows = []
    for run_path in compare_runs:
        shortlist_df = load_run_table_safe(run_path, "candidate_shortlist.parquet")
        manifest = load_manifest(run_path)
        compare_rows.append(
            {
                "run_path": run_path,
                "benchmark_name": manifest.get("benchmark_name"),
                "signals_path": manifest.get("signals_path"),
                "best_total_pnl": safe_metric(shortlist_df, "total_pnl"),
                "best_max_drawdown": safe_metric(shortlist_df, "max_drawdown"),
                "best_profit_factor": safe_metric(shortlist_df, "profit_factor"),
                "best_avg_hold_minutes": safe_metric(shortlist_df, "avg_hold_minutes"),
            }
        )
    compare_df = pd.DataFrame(compare_rows)
    if not compare_df.empty:
        st.subheader("Benchmark Comparison")
        st.dataframe(compare_df, width="stretch")
        st.plotly_chart(px.bar(compare_df, x="benchmark_name", y="best_total_pnl", color="run_path", title="Best Total PnL by Run"), width="stretch")
        long_df = compare_df.melt(
            id_vars=["run_path", "benchmark_name", "signals_path"],
            value_vars=[col for col in ["best_total_pnl", "best_max_drawdown", "best_profit_factor", "best_avg_hold_minutes"] if col in compare_df.columns],
            var_name="metric",
            value_name="value",
        )
        st.plotly_chart(px.bar(long_df, x="benchmark_name", y="value", color="run_path", facet_col="metric", title="Compare benchmark runs by metric"), width="stretch")
