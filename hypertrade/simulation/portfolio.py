"""Fixed maximum position sizing and 100% short collateral; no allocation optimizer."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


class CapitalLedger:
    def __init__(self, params):
        self.available = float(params.initial_capital)
        self.limit = float(params.maximum_position_notional)
        self.active: list[dict] = []

    def settle(self, timestamp):
        remaining = []
        for trade in self.active:
            if trade['exit_event_dt'] <= timestamp:
                self.available += trade['reserved_notional'] + trade['gross_pnl_usd'] - trade['exit_fee_usd']
            else:
                remaining.append(trade)
        self.active = remaining

    def admit(self, source: dict) -> dict | None:
        px, fee = source['entry_price'], source['commission_per_share_side']
        if px <= 0 or not math.isfinite(px):
            raise ValueError('Entry price must be finite and positive')
        # The $10k cap is gross notional. Available cash also covers the entry fee.
        quantity = math.floor(min(self.limit / px, max(0.0, self.available) / (px + fee)))
        if quantity <= 0:
            return None
        trade = dict(source)
        trade.update(quantity=quantity, reserved_notional=quantity*px,
                     entry_fee_usd=quantity*fee, exit_fee_usd=quantity*fee,
                     gross_pnl_usd=quantity*(source['exit_price']-px)*source['direction'],
                     cash_pnl_usd=quantity*source['pnl'])
        self.available -= trade['reserved_notional'] + trade['entry_fee_usd']
        if self.available < -1e-7:
            raise ArithmeticError('Admission created negative buying power')
        self.active.append(trade)
        return trade


def portfolio_equity(trades: list[dict], marks: dict[int, pd.Series], initial: float) -> tuple[pd.DataFrame, dict]:
    """Settle intrabar exits, mark prior closes, settle open exits, then enter."""
    columns = ['datetime', 'equity_usd', 'realized_pnl_usd', 'unrealized_pnl_usd',
               'available_cash_usd', 'buying_power_usd', 'reserved_notional_usd',
               'long_security_value_usd', 'short_collateral_usd', 'short_liability_usd', 'locked_short_proceeds_usd', 'drawdown_pct']
    if not trades:
        return pd.DataFrame(columns=columns), {'portfolio_return_pct': 0.0, 'portfolio_max_drawdown_pct': 0.0,
                                               'portfolio_pnl_usd': 0.0, 'portfolio_trade_count': 0.0}
    start = min(t['entry_dt'] for t in trades)
    end = max(t['exit_event_dt'] for t in trades)
    timeline = {start, end}
    entries, exits = {}, {}
    for i, t in enumerate(trades):
        entries.setdefault(t['entry_dt'], []).append(i)
        exits.setdefault(t['exit_event_dt'], []).append(i)
        timeline.update([t['entry_dt'], t['exit_event_dt']])
        timeline.update(marks[t['candidate_id']].index)
    # Union of actual 15-minute candle closes, no invented prices or overnight interpolation.
    active: set[int] = set()
    realized, peak = 0.0, initial
    rows = []
    mark_arrays = {i: (s.index.as_unit("ns").asi8, s.to_numpy(float)) for i, s in marks.items()}

    def snapshot(ts, phase):
        nonlocal peak
        unrealized = reserve = long_value = short_value = short_proceeds = fees = 0.0
        for i in sorted(active):
            t = trades[i]
            times, prices = mark_arrays[t['candidate_id']]
            k = int(np.searchsorted(times, ts.value, side='right')) - 1
            px = prices[k] if k >= 0 else t['entry_price']
            q = t['quantity']
            unrealized += (px-t['entry_price'])*q*t['direction']-t['entry_fee_usd']
            fees += t['entry_fee_usd']
            reserve += t['reserved_notional']
            if t['direction'] == 1:
                long_value += px*q
            else:
                short_value += px*q
                short_proceeds += t['reserved_notional']
        equity = initial+realized+unrealized
        available = initial+realized-reserve-fees
        peak = max(peak, equity)
        rows.append(dict(zip(columns, [ts, equity, realized, unrealized, available, max(0.0, available), reserve,
                                        long_value, short_proceeds, short_value, short_proceeds, (peak-equity)/peak*100 if peak > 0 else 0.0]), event_phase=phase))

    snapshot(start, 'initial')
    for ts in sorted(t for t in timeline if start <= t <= end):
        for i in sorted(exits.get(ts, [])):
            if i in active and trades[i]['exit_dt'] < ts:
                active.remove(i)
                realized += trades[i]['cash_pnl_usd']
        if active or exits.get(ts):
            # An open/gap exit may still have a genuine preceding candle close
            # at this timestamp. Intrabar executions have already settled.
            snapshot(ts, 'close_marks_after_intrabar_exits')
        for i in sorted(exits.get(ts, [])):
            if i in active:
                active.remove(i)
                realized += trades[i]['cash_pnl_usd']
        for i in sorted(entries.get(ts, [])):
            active.add(i)
        # Zero-duration open exits occur after the whole entry batch, without same-batch reuse.
        for i in sorted(exits.get(ts, [])):
            if i in active and trades[i]['entry_dt'] == ts:
                active.remove(i)
                realized += trades[i]['cash_pnl_usd']
        if entries.get(ts) or exits.get(ts):
            snapshot(ts, 'after_exits_then_entries')
    frame = pd.DataFrame(rows)
    final = float(frame.iloc[-1]['equity_usd'])
    return frame, {'portfolio_return_pct': (final/initial-1)*100,
                   'portfolio_max_drawdown_pct': float(frame['drawdown_pct'].max()),
                   'portfolio_pnl_usd': final-initial, 'portfolio_trade_count': float(len(trades))}
