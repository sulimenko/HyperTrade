# Research terminal

Start `.venv/bin/python run_dashboard.py`. The default theme and all Plotly charts use navy surfaces, cyan and violet accents with light text. There are five sidebar views; a shared experiment selector connects them.

## Research Home

Search the experiment list and inspect latest study/baseline status. Empty workspaces display instructions, without fabricated samples. A study with no valid candidates is a normal result.

## New Experiment

Choose a fixed benchmark, custom CSV path or uploaded CSV. Configure Pareto objectives/directions, constraints and preferred-candidate policy. Search toggles and costs override corresponding values in the full JSON editor; all other editable search fields are passed through. Units for research sums, portfolio metrics, per-share commission and fractional slippage are explicit.

Start runs one baseline and the configured trials. Profiles are loaded from `config/defaults/profiles/*.json` or edited inline. Existing run profiles/artifact hashes are never rewritten by this view. Failed runs expose persisted diagnostics.

## Optimizer Explorer

Choose the Pareto frontier or all trials with metric values. Filter COMPLETE/PRUNED/FAIL. Assign arbitrary numeric x/y/z and color axes; 3D supports rotation/zoom. Since Plotly 3D selection is limited, use the linked candidate selector or selectable Pareto table. The 2D plot supports point/box/lasso selection. The selected Pareto candidate is shared with Candidate Analysis.

A session shortlist starts with the optimizer's deterministic preferred candidates and supports adding/removing Pareto trials. It persists while navigating this browser session, not across new browser sessions; stored experiment artifacts remain unchanged. Complete metrics, trades and equity are persisted for every Pareto candidate. Non-Pareto trial summaries, parameters, constraints and diagnostics remain inspectable in the trials tab.

## Candidate Analysis

Switch between Research / Normalized and Capital / Portfolio Equity. Three primary metrics compare the candidate to the fixed unfiltered baseline. Tabs disclose curves, distributions, trades, final rejection counts/details, settings and execution assumptions. Normalized sequential PnL/proxy drawdown is labelled separately from chronological money equity/percent drawdown.

## Compare & Validation

Compare stored baseline and shortlisted outcomes between runs; inspect source hashes, model assumptions and objective profiles before interpreting differences. Run both fixed benchmark fixtures or inspect earlier benchmark suites. No walk-forward, OOS readiness guarantee or predictive validation is claimed.
