from __future__ import annotations

from pathlib import Path

import streamlit as st

from hypertrade.config import DEFAULT_FILTER_SEARCH_SPACE, FilterSearchSpace, ObjectiveProfile, OptimizationRunConfig
from hypertrade.optimization import run_optimization
from hypertrade.ui.state import ARTIFACT_ROOT, DEFAULT_BENCHMARKS, PROFILE_ROOT, SEARCH_SPACE_ROOT


def _search_space_form(default: FilterSearchSpace) -> FilterSearchSpace:
    st.subheader("Search Space")
    col1, col2, col3 = st.columns(3)
    sl_min = col1.number_input("SL min", value=float(default.sl_min), step=0.5)
    sl_max = col2.number_input("SL max", value=float(default.sl_max), step=0.5)
    sl_step = col3.number_input("SL step", value=float(default.sl_step), step=0.25)

    tp_col1, tp_col2, tp_col3 = st.columns(3)
    tp_min = tp_col1.number_input("TP min", value=float(default.tp_min), step=0.5)
    tp_max = tp_col2.number_input("TP max", value=float(default.tp_max), step=0.5)
    tp_step = tp_col3.number_input("TP step", value=float(default.tp_step), step=0.25)

    delay_col1, delay_col2, delay_col3 = st.columns(3)
    delay_open_min = delay_col1.number_input("Delay min", value=int(default.delay_open_min), step=30)
    delay_open_max = delay_col2.number_input("Delay max", value=int(default.delay_open_max), step=30)
    delay_open_step = delay_col3.number_input("Delay step", value=int(default.delay_open_step), step=30)

    hold_col1, hold_col2, hold_col3 = st.columns(3)
    holding_minutes_min = hold_col1.number_input("Hold min", value=int(default.holding_minutes_min), step=60)
    holding_minutes_max = hold_col2.number_input("Hold max", value=int(default.holding_minutes_max), step=60)
    holding_minutes_step = hold_col3.number_input("Hold step", value=int(default.holding_minutes_step), step=60)

    confirm_col1, confirm_col2 = st.columns(2)
    confirm_bars_min = confirm_col1.number_input("Confirm bars min", value=int(default.confirm_bars_min), step=1)
    confirm_bars_max = confirm_col2.number_input("Confirm bars max", value=int(default.confirm_bars_max), step=1)

    toggle_cols = st.columns(5)
    ema_use = toggle_cols[0].checkbox("EMA", value=bool(default.ema_use))
    rsi_use = toggle_cols[1].checkbox("RSI", value=bool(default.rsi_use))
    adx_use = toggle_cols[2].checkbox("ADX", value=bool(default.adx_use))
    macd_use = toggle_cols[3].checkbox("MACD", value=bool(default.macd_use))
    bb_use = toggle_cols[4].checkbox("BB", value=bool(default.bb_use))

    toggle_cols2 = st.columns(5)
    atr_use = toggle_cols2[0].checkbox("ATR", value=bool(default.atr_use))
    donchian_use = toggle_cols2[1].checkbox("Donchian", value=bool(default.donchian_use))
    psar_use = toggle_cols2[2].checkbox("PSAR", value=bool(default.psar_use))
    ts_use = toggle_cols2[3].checkbox("TS", value=bool(default.ts_use))
    vwap_use = toggle_cols2[4].checkbox("VWAP", value=bool(default.vwap_use))
    volume_use = st.checkbox("Volume", value=bool(default.volume_use))

    acceptance_cols = st.columns(4)
    time_gate_use = acceptance_cols[0].checkbox("Time gate", value=bool(default.time_gate_use))
    cooldown_use = acceptance_cols[1].checkbox("Cooldown", value=bool(default.cooldown_use))
    max_signals_per_side_max = acceptance_cols[2].number_input("Max signals/side", value=int(default.max_signals_per_side_max), step=1)
    ranking_mode_choices = st.multiselect("Ranking modes", ["none", "strength"], default=default.ranking_mode_choices)

    time_cols = st.columns(4)
    accept_start_min = time_cols[0].number_input("Accept start min", value=int(default.accept_start_min), step=30)
    accept_start_max = time_cols[1].number_input("Accept start max", value=int(default.accept_start_max), step=30)
    accept_end_min = time_cols[2].number_input("Accept end min", value=int(default.accept_end_min), step=30)
    accept_end_max = time_cols[3].number_input("Accept end max", value=int(default.accept_end_max), step=30)

    cooldown_cols = st.columns(3)
    cooldown_min = cooldown_cols[0].number_input("Cooldown min", value=int(default.cooldown_min), step=30)
    cooldown_max = cooldown_cols[1].number_input("Cooldown max", value=int(default.cooldown_max), step=30)
    cooldown_step = cooldown_cols[2].number_input("Cooldown step", value=int(default.cooldown_step), step=30)

    return FilterSearchSpace(
        sl_min=float(sl_min),
        sl_max=float(sl_max),
        sl_step=float(sl_step),
        tp_min=float(tp_min),
        tp_max=float(tp_max),
        tp_step=float(tp_step),
        delay_open_min=int(delay_open_min),
        delay_open_max=int(delay_open_max),
        delay_open_step=int(delay_open_step),
        holding_minutes_min=int(holding_minutes_min),
        holding_minutes_max=int(holding_minutes_max),
        holding_minutes_step=int(holding_minutes_step),
        confirm_bars_min=int(confirm_bars_min),
        confirm_bars_max=int(confirm_bars_max),
        ema_use=bool(ema_use),
        rsi_use=bool(rsi_use),
        adx_use=bool(adx_use),
        macd_use=bool(macd_use),
        bb_use=bool(bb_use),
        atr_use=bool(atr_use),
        donchian_use=bool(donchian_use),
        psar_use=bool(psar_use),
        ts_use=bool(ts_use),
        vwap_use=bool(vwap_use),
        volume_use=bool(volume_use),
        time_gate_use=bool(time_gate_use),
        cooldown_use=bool(cooldown_use),
        accept_start_min=int(accept_start_min),
        accept_start_max=int(accept_start_max),
        accept_end_min=int(accept_end_min),
        accept_end_max=int(accept_end_max),
        cooldown_min=int(cooldown_min),
        cooldown_max=int(cooldown_max),
        cooldown_step=int(cooldown_step),
        max_signals_per_side_max=int(max_signals_per_side_max),
        ranking_mode_choices=ranking_mode_choices or ["none"],
        commission=float(default.commission),
        slippage=float(default.slippage),
        bar_minutes=int(default.bar_minutes),
    )


def render_launcher() -> None:
    st.header("Run Launcher")

    benchmark_choice = st.selectbox("Benchmark", list(DEFAULT_BENCHMARKS.keys()) + ["custom"])
    default_signals = DEFAULT_BENCHMARKS.get(benchmark_choice, "benchmarks/fixtures/PF20250597.csv")
    signals_path = st.text_input(
        "Signals path",
        value=default_signals if benchmark_choice != "custom" else "benchmarks/fixtures/PF20250597.csv",
    )
    benchmark_name = benchmark_choice if benchmark_choice != "custom" else Path(signals_path).stem
    run_name = st.text_input("Run label", value=Path(signals_path).stem)
    n_trials = st.number_input("Trials", min_value=1, value=10, step=1)
    st.caption(f"Benchmark identity: {benchmark_name}")

    profile_files = sorted(PROFILE_ROOT.glob("*.json"))
    if not profile_files:
        st.warning("No saved objective profiles found. Create one on the Objectives page first.")
        return
    selected_profile = st.selectbox("Objective profile", [path.name for path in profile_files])
    profile_path = PROFILE_ROOT / selected_profile
    objective_profile = ObjectiveProfile.load(profile_path)
    st.caption(f"Using profile: {objective_profile.name}")

    search_space_files = sorted(SEARCH_SPACE_ROOT.glob("*.json"))
    search_source = st.selectbox("Search space source", ["inline editor"] + [path.name for path in search_space_files])
    default_search_space = DEFAULT_FILTER_SEARCH_SPACE
    if search_source != "inline editor":
        default_search_space = FilterSearchSpace.load(SEARCH_SPACE_ROOT / search_source)

    search_space_name = st.text_input("Search space name", value="default_search_space")
    search_space = _search_space_form(default_search_space)

    action_cols = st.columns(2)
    if action_cols[0].button("Save search space", width="stretch"):
        search_path = SEARCH_SPACE_ROOT / f"{search_space_name}.json"
        search_space.save(search_path)
        st.success(f"Saved {search_path}")

    if action_cols[1].button("Run optimization", width="stretch"):
        with st.spinner("Running optimization..."):
            run_dir = run_optimization(
                OptimizationRunConfig(
                    signals_path=signals_path,
                    n_trials=int(n_trials),
                    benchmark_name=benchmark_name,
                    run_label=run_name,
                    objective_profile_path=str(profile_path),
                    study_name=run_name,
                    artifacts_root=str(ARTIFACT_ROOT),
                ),
                objective_profile=objective_profile,
                search_space=search_space,
            )
        st.success(f"Completed run: {run_dir}")
