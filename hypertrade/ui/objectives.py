from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from hypertrade.config import DEFAULT_OBJECTIVE_PROFILE, OBJECTIVE_CATALOG
from hypertrade.config.schemas import CANDIDATE_POLICY_MODES
from hypertrade.ui.state import PROFILE_ROOT, load_json_file, load_objective_profile_for_run


def _constraints_frame(current: dict) -> pd.DataFrame:
    rows = current.get("constraints", [])
    if not rows:
        rows = [
            {"metric": "trades", "min_value": 20.0, "max_value": None},
            {"metric": "profit_factor", "min_value": 1.05, "max_value": None},
        ]
    return pd.DataFrame(rows)


def _constraint_payload(frame: pd.DataFrame) -> list[dict]:
    payload = []
    for row in frame.fillna("").to_dict(orient="records"):
        metric = str(row.get("metric", "")).strip()
        if not metric:
            continue
        min_value = row.get("min_value", "")
        max_value = row.get("max_value", "")
        payload.append(
            {
                "metric": metric,
                "min_value": None if min_value == "" else float(min_value),
                "max_value": None if max_value == "" else float(max_value),
            }
        )
    return payload


def _profile_payload(
    profile_name: str,
    selected_metrics: list[str],
    directions: dict[str, str],
    constraints_df: pd.DataFrame,
    candidate_policy_mode: str,
) -> dict:
    return {
        "name": profile_name,
        "objectives": [{"metric": metric, "direction": directions[metric]} for metric in selected_metrics],
        "constraints": _constraint_payload(constraints_df),
        "candidate_policy": {"mode": candidate_policy_mode},
    }


def render_objectives(selected_run: str | None) -> None:
    st.header("Objective Editor")
    if not selected_run:
        st.info("Select a run to view or save a run-local objective profile.")
        return

    current = load_objective_profile_for_run(selected_run)
    profile_files = sorted(PROFILE_ROOT.glob("*.json"))
    selected_profile_file = st.selectbox("Load saved profile", ["<current run profile>"] + [path.name for path in profile_files])
    if selected_profile_file != "<current run profile>":
        current = load_json_file(PROFILE_ROOT / selected_profile_file, DEFAULT_OBJECTIVE_PROFILE.to_dict())

    profile_name = st.text_input("Profile name", value=current.get("name", "ui_profile"))
    selected_metrics = st.multiselect(
        "Objectives",
        list(OBJECTIVE_CATALOG.keys()),
        default=[item["metric"] for item in current.get("objectives", [])],
    )

    current_directions = {item["metric"]: item["direction"] for item in current.get("objectives", [])}
    directions = {}
    for metric in selected_metrics:
        default_direction = current_directions.get(metric, OBJECTIVE_CATALOG[metric]["direction"])
        directions[metric] = st.selectbox(
            f"{metric} direction",
            ["maximize", "minimize"],
            index=0 if default_direction == "maximize" else 1,
            key=f"direction_{metric}",
        )

    candidate_policy_mode = st.selectbox(
        "Preferred candidate policy",
        sorted(CANDIDATE_POLICY_MODES),
        index=sorted(CANDIDATE_POLICY_MODES).index(
            current.get("candidate_policy", {}).get("mode", "utopia_distance")
            if current.get("candidate_policy", {}).get("mode", "utopia_distance") in CANDIDATE_POLICY_MODES
            else "utopia_distance"
        ),
    )

    st.subheader("Constraints")
    constraints_df = st.data_editor(_constraints_frame(current), num_rows="dynamic", width="stretch", hide_index=True)

    if st.button("Save objective profile", width="stretch"):
        if not selected_metrics:
            st.error("Select at least one objective before saving.")
            return
        payload = _profile_payload(profile_name, selected_metrics, directions, constraints_df, candidate_policy_mode)
        profile_path = PROFILE_ROOT / f"{profile_name}.json"
        profile_path.write_text(json.dumps(payload, indent=2))
        Path(selected_run, "objective_profile.json").write_text(json.dumps(payload, indent=2))
        st.success(f"Saved {profile_path}")

    st.subheader("Current payload")
    st.json(_profile_payload(profile_name, selected_metrics, directions, constraints_df, candidate_policy_mode))
    if profile_files:
        st.subheader("Saved profiles")
        st.write([str(path) for path in profile_files])
    st.subheader("Metric Catalog")
    st.dataframe(pd.DataFrame(OBJECTIVE_CATALOG).T, width="stretch")
