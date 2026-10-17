from __future__ import annotations
import math
import pandas as pd
from hypertrade.config.schemas import ObjectiveProfile

def pick_preferred_candidates(df: pd.DataFrame, objective_profile: ObjectiveProfile, limit: int = 5) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    frame = df.copy()
    if "constraint_violations" in frame.columns:
        constraint_series = frame["constraint_violations"].fillna("").astype(str)
        frame = frame[constraint_series.eq("")]
    if frame.empty:
        return frame

    for objective in objective_profile.objectives:
        if objective.metric not in frame.columns:
            return frame.iloc[0:0].copy()
        column = f"metric_{objective.metric}"
        frame[column] = pd.to_numeric(frame[objective.metric], errors="coerce")
        series = frame[column]
        lo = series.min()
        hi = series.max()
        denom = (hi - lo) if pd.notna(hi) and pd.notna(lo) and hi != lo else 1.0
        if objective.direction == "maximize":
            frame[f"norm_{objective.metric}"] = (hi - series) / denom
        else:
            frame[f"norm_{objective.metric}"] = (series - lo) / denom

    score = 0.0
    for objective in objective_profile.objectives:
        score = score + frame[f"norm_{objective.metric}"].pow(2)
    frame["utopia_distance"] = score.apply(lambda x: math.sqrt(float(x)))

    mode = objective_profile.candidate_policy.normalized_mode()
    if mode == "max_sum_trade_returns_pct" and "sum_trade_returns_pct" in frame.columns:
        return frame.sort_values(["sum_trade_returns_pct", "utopia_distance"], ascending=[False, True]).head(limit)
    if mode == "max_profit_factor" and "profit_factor" in frame.columns:
        return frame.sort_values(["profit_factor", "utopia_distance"], ascending=[False, True]).head(limit)
    if mode == "min_proxy_drawdown_usd" and "proxy_drawdown_usd" in frame.columns:
        return frame.sort_values(["proxy_drawdown_usd", "utopia_distance"], ascending=[True, True]).head(limit)
    if mode == "min_avg_hold_minutes" and "avg_hold_minutes" in frame.columns:
        return frame.sort_values(["avg_hold_minutes", "utopia_distance"], ascending=[True, True]).head(limit)
    return frame.sort_values("utopia_distance", ascending=True).head(limit)


def constraint_violations(metrics: dict, objective_profile: ObjectiveProfile) -> list[str]:
    violations = []
    for rule in objective_profile.constraints:
        value = metrics.get(rule.metric)
        if value is None or not math.isfinite(float(value)):
            violations.append(f"missing:{rule.metric}")
            continue
        if rule.min_value is not None and float(value) < float(rule.min_value):
            violations.append(f"{rule.metric}<min")
        if rule.max_value is not None and float(value) > float(rule.max_value):
            violations.append(f"{rule.metric}>max")
    return violations


def _objective_values(metrics: dict, objective_profile: ObjectiveProfile) -> tuple[float, ...]:
    return tuple(float(metrics[objective.metric]) for objective in objective_profile.objectives)


