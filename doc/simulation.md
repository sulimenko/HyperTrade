# Execution, admission and money semantics

## Trade exits and costs

For long PSAR crossings, execution uses `min(SAR, open)`; short uses `max(SAR, open)`. The extreme point updates the indicator, never the fill. PSAR initializes from the completed entry candle high/low and starts checking the next candle. The known initial trailing level and SL/TP are checked on the entry candle; holding zero bars is valid. That candle is traversed once. A requested zero market-minute hold exits at its entry open and pays both commissions/slippage.

The user intentionally retains two assumptions: filters/ranking can see the entry candle's OHLC when entering at its open, and the existing same-bar SL/TP heuristic remains. For a bullish candle (`close >= open`), long chooses SL first and short TP first; for a bearish candle, long chooses TP first and short SL first. These are research assumptions, not review blockers. Later trailing stops retain the existing high/low update heuristic. Time exit precedes PSAR, trailing, SL/TP.

Entry fill is `open * (1 + direction * slippage)`, exit fill is `raw_exit * (1 - direction * slippage)`. Direction is +1 long / -1 short. Net PnL per share is `(exit-entry)*direction - 2*commission`. Commission is USD/share/side; slippage is a dimensionless fraction/side. If data ends before the holding deadline, exit at the last close and retain `data_truncated=true` in trade artifacts.

## Final admission

Time/data/indicator eligibility precedes a non-mutating cooldown check, ranking, side cap and affordable admission. Unexpected calculation errors propagate, never become rejection reasons. ATR/technical rejection cannot reserve capital or update cooldown. Ranked candidates unable to buy one share generate `rejected_insufficient_capital`; the next eligible candidate can fill the side cap. A rejected signal never changes cooldown. Counts and per-symbol reasons are saved separately for both models.

The actual entry candle time, rather than an off-grid CSV timestamp, determines chronology and cooldown. Same entry-time candidates sort by descending ranking, ties by stable input order; grouped CSV rows retain their original symbol order. Long and short compete for the same budget with that deterministic order, without an allocation optimizer.

At each timestamp, available close marks are recorded first, existing exits settle, then new entries reserve cash. Intrabar exits settle at the candle close; gap/time exits settle at the open. New zero-duration exits settle after the entire entry batch, so their released cash is not reused within that same batch. Adjacent batches can reuse settled cash. Long and short keep separate side caps and a shared per-symbol cooldown.

## Research / Normalized

Every admitted trade has $10,000 nominal divided by its slipped entry price (fractional analytical exposure). Net normalized money PnL is `10000 * return_pct / 100`. PF uses positive versus absolute negative net normalized PnL. With no losing trades PF is undefined (`null`), rather than an arbitrary capped number; an objective or constraint requiring an undefined PF prunes that candidate normally.

The sequential curve orders by entry time, symbol and direction, accumulates normalized PnL and includes a zero-PnL initial peak for drawdown. It ignores overlap/funding and is labelled a **proxy**. `sum_trade_returns_pct` sums percentage points; it is never labelled portfolio return. Mean/median, win rate, count, average holding minutes and left-tail loss CVaR use net trade outcomes. Loss CVaR is the nonnegative opposite of the worst ceil(5% × count) mean return, in percent.

The fixed baseline disables all indicator filters and admission rules. Its SL/TP/delay/holding settings use search minima, with the same commission and slippage as candidates. Comparisons are fixed-reference improvements and can include exit tuning; they do not isolate the causal effect of filters alone.

## Capital / Portfolio Equity

Initial equity is $100,000; fixed maximum position notional is $10,000. Quantity is the smaller integer of `floor(10000 / entry_price)` and `floor(available_cash / (entry_price + entry_commission))`. No positive affordable quantity means rejection. Entry fee and notional/collateral are reserved only on final acceptance. Neither sizing constant is an Optuna parameter.

A long position reserves its purchase notional. A short reserves 100% entry notional as collateral and keeps sale proceeds locked. Short proceeds are not buying power. Closing releases the reservation and realizes directional gross PnL minus exit fee; entry fee was already debited. Collateral and locked proceeds are separate ledger columns, even though their initial dollar amounts match.

Equity is `initial + realized_net_PnL + active_gross_mark_PnL - active_entry_fees`. Equivalently, equity is available cash plus long securities plus short collateral plus locked short proceeds minus the current short liability. This prevents double counting sale proceeds. Return is `(final_equity/initial - 1)*100`; drawdown is `(peak-equity)/peak*100`, including initial equity as a peak.

Mark-to-market uses the union of actual 15-minute candle closes, entry and settlement events. A close becomes visible at candle open +15 minutes, never at that candle's open. Newly opened positions are first marked at the raw open, so entry slippage is reflected immediately. The last available close is carried across absent bars. Unrealized liquidation commissions/slippage are not accrued; actual exit costs are charged on settlement. There is no maintenance-margin/liquidation model, borrow fee, borrow availability or corporate-action model. A catastrophic short gap can create a negative cash balance/equity after settlement; buying power stays zero and no new reservation is allowed. Gap losses are never silently clipped.
