"""One eligibility pipeline, independent normalized and capital-limited admissions."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import json
import pandas as pd

from hypertrade.data.signals import utc_timestamp
from hypertrade.features.indicators import ensure_market_data
from hypertrade.filters.acceptance import passes_time_gate
from hypertrade.filters.ranking import rank_score
from hypertrade.simulation.market_time import build_market_cache, compute_entry_time_cached
from hypertrade.simulation.trade_engine import simulate_trade
from hypertrade.simulation.metrics import compute_metrics, sequential_curve
from hypertrade.simulation.portfolio import CapitalLedger, portfolio_equity


@dataclass
class BacktestResult:
    trades: list[dict]
    signal_stats: list[dict]
    metrics: dict
    portfolio_trades: list[dict]
    portfolio_signal_stats: list[dict]
    equity: pd.DataFrame
    sequential: pd.DataFrame


def evaluate(signals: list[dict], params) -> BacktestResult:
    if not signals:
        equity, capital_metrics = portfolio_equity([], {}, params.initial_capital)
        return BacktestResult([], [], {**compute_metrics([], params), **capital_metrics}, [], [], equity, sequential_curve([]))
    ordered = sorted(signals, key=lambda s: utc_timestamp(s['datetime']))
    cache = build_market_cache(utc_timestamp(ordered[0]['datetime']), utc_timestamp(ordered[-1]['datetime']),
                               extra_days=max(60, int((params.holding_minutes+params.delay_open)/390)*2+30))
    candidates, rejections, marks = [], [], {}
    candidate_id = 0
    for signal in ordered:
        for label, direction in [('long', 1), ('short', -1)]:
            for symbol in dict.fromkeys(signal.get(label, [])):
                stamp = utc_timestamp(signal['datetime'])
                def reject(reason):
                    rejections.append({'datetime': stamp, 'symbol': symbol, 'direction': direction, 'reason': reason})
                if not passes_time_gate(stamp, params):
                    reject('rejected_time_gate')
                    continue
                frame = ensure_market_data(symbol, stamp, params.indicator_config)
                if frame is None or frame.empty:
                    reject('rejected_no_data')
                    continue
                expected = pd.Timestamp(compute_entry_time_cached(stamp, params.delay_open, cache))
                idx = int(frame['datetime'].searchsorted(expected))
                if idx >= len(frame):
                    reject('rejected_no_candles_after_entry')
                    continue
                # Technical exits/ATR eligibility must succeed before any reservation/cooldown mutation.
                trade = simulate_trade(symbol, stamp, params, frame, direction, cache)
                if trade.get('rejected'):
                    reject('rejected_'+trade['reject_reason'])
                    continue
                trade.update(candidate_id=candidate_id, signal_dt=stamp,
                             rank_score=rank_score(frame, idx, params, direction))
                end_idx = int(frame['datetime'].searchsorted(trade['exit_dt'], side='right'))
                candles = frame.iloc[idx:end_idx]
                close_times = pd.DatetimeIndex(candles['datetime']) + pd.Timedelta(minutes=params.bar_minutes)
                marks[candidate_id] = pd.Series([trade['entry_mark_price'], *candles['close'].to_numpy(float)],
                                                index=pd.DatetimeIndex([trade['entry_dt'], *close_times]))
                candidates.append(trade)
                candidate_id += 1

    def select(capital: bool):
        ledger = CapitalLedger(params) if capital else None
        last_entry = {}
        accepted, details = [], list(rejections)
        groups = {}
        for t in candidates:
            groups.setdefault(t['entry_dt'], []).append(t)
        for timestamp in sorted(groups):
            if ledger:
                ledger.settle(timestamp)
            counts = Counter()
            # Stable original CSV order for equal scores; long then short within a grouped signal.
            eligible = []
            for t in groups[timestamp]:
                previous = last_entry.get(t['symbol'])
                if previous is not None and (timestamp-previous).total_seconds()/60 < params.cooldown_minutes:
                    details.append(dict(datetime=timestamp, symbol=t['symbol'], direction=t['direction'], reason='rejected_cooldown'))
                else:
                    eligible.append(t)
            eligible.sort(key=lambda t: (-t['rank_score'], t['candidate_id']))
            for t in eligible:
                reason = None
                previous = last_entry.get(t['symbol'])
                if previous is not None and (timestamp-previous).total_seconds()/60 < params.cooldown_minutes:
                    reason = 'rejected_cooldown'
                elif params.max_signals_per_side is not None and counts[t['direction']] >= params.max_signals_per_side:
                    reason = 'rejected_side_cap'
                trade = None
                if reason is None:
                    trade = ledger.admit(t) if ledger else dict(t)
                    if trade is None:
                        reason = 'rejected_insufficient_capital'
                if reason:
                    details.append(dict(datetime=timestamp, symbol=t['symbol'], direction=t['direction'], reason=reason))
                    continue
                if not capital:
                    trade['normalized_pnl_usd'] = trade['return_pct']*params.normalized_notional/100
                accepted.append(trade)
                counts[t['direction']] += 1
                last_entry[t['symbol']] = timestamp
        stats = []
        times = sorted({utc_timestamp(s['datetime']) for s in signals} | {t['entry_dt'] for t in candidates} | {d['datetime'] for d in details})
        for stamp in times:
            rejects = [d for d in details if d['datetime'] == stamp]
            admitted = [t for t in accepted if t['entry_dt'] == stamp]
            reasons = Counter(d['reason'] for d in rejects)
            stats.append(dict(datetime=stamp, model='portfolio' if capital else 'research',
                              symbols_total=len(admitted)+len(rejects), symbols_traded=len(admitted), symbols_rejected=len(rejects),
                              accepted_long=sum(t['direction'] == 1 for t in admitted),
                              accepted_short=sum(t['direction'] == -1 for t in admitted),
                              rejection_details=json.dumps(rejects, default=str), **reasons))
        return accepted, stats

    research, research_stats = select(False)
    portfolio, portfolio_stats = select(True)
    equity, portfolio_metrics = portfolio_equity(portfolio, marks, params.initial_capital)
    return BacktestResult(research, research_stats, {**compute_metrics(research, params), **portfolio_metrics},
                          portfolio, portfolio_stats, equity, sequential_curve(research, params.normalized_notional))
