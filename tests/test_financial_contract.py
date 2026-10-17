"""Two focused regression scenarios for approved execution and money contracts."""
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from hypertrade.config.strategy import StrategyParams
from hypertrade.simulation.trade_engine import simulate_trade
from hypertrade.simulation.backtest import evaluate


def market(rows):
    return pd.DataFrame(rows,columns=['open','high','low','close']).assign(
        datetime=pd.date_range('2025-07-10 14:00',periods=len(rows),freq='15min',tz='UTC'))


def test_execution_prices_entry_bar_and_per_share_costs():
    params=StrategyParams(sl=20,tp=20,holding_minutes=60,commission=.02,slippage=0,
                          psar_enabled=True,psar_step=.02,psar_max=.2)
    for direction,second,price in [(1,[100,101,97,99],98),(1,[96,97,94,95],96),
                                    (-1,[100,103,99,101],102),(-1,[104,106,103,105],104)]:
        frame=market([[100,102,98,100],second])
        trade=simulate_trade('AAA',frame.iloc[0].datetime,params,frame,direction)
        assert trade['exit_reason']=='psar'
        assert trade['exit_price']==price
        assert trade['hold_bars']==1
        assert trade['pnl']==pytest.approx((price-100)*direction-.04)
    # Preserve the bullish/bearish intrabar choice even when both SL and TP hit.
    for direction,close,price in [(1,101,98),(1,99,103),(-1,101,97),(-1,99,102)]:
        frame=market([[100,105,95,close],[100,101,99,100]])
        trade=simulate_trade('AAA',frame.iloc[0].datetime,replace(params,psar_enabled=False,sl=2,tp=3),frame,direction)
        assert trade['exit_price']==price
        assert trade['hold_bars']==0
    frame=market([[100,104,94,101],[100,101,99,100]])
    trade=simulate_trade('AAA',frame.iloc[0].datetime,replace(params,psar_enabled=False,ts_enabled=True,ts_dist=5),frame)
    assert trade['exit_price']==95
    assert trade['hold_bars']==0
    # Entry fill and exit fill each pay directional slippage, fees stay USD/share/side.
    frame=market([[100,110,90,101]])
    for direction in [1,-1]:
        trade=simulate_trade('AAA',frame.iloc[0].datetime,replace(params,psar_enabled=False,sl=2,tp=3,slippage=.001),frame,direction)
        assert trade['entry_price']==pytest.approx(100*(1+direction*.001))
        assert trade['pnl']==pytest.approx((trade['exit_price']-trade['entry_price'])*direction-.04)


def test_admission_capital_backfill_cooldown_and_overlapping_equity():
    frames={
        'A':market([[60,64,59,63],[60,61,59,60],[60,61,59,60]]),
        'B':market([[100,104,99,103],[30,32,29.9,31],[30,31,29,30]]),
        'C':market([[10,10.4,9.9,10.3],[10,10.2,9.8,10],[10,10.2,9.8,10]]),
    }
    start=frames['A'].iloc[0].datetime
    signals=[dict(datetime=start,long=['A','B','C'],short=[]),
             dict(datetime=start+pd.Timedelta(minutes=15),long=['B'],short=[])]
    params=StrategyParams(sl=10,tp=3,commission=.02,slippage=0,initial_capital=100,
                          maximum_position_notional=60,max_signals_per_side=2,cooldown_minutes=30)
    with patch('hypertrade.simulation.backtest.ensure_market_data',side_effect=lambda symbol,*args:frames[symbol]):
        result=evaluate(signals,params)
    assert [t['symbol'] for t in result.trades]==['A','B']
    assert [t['symbol'] for t in result.portfolio_trades]==['A','C','B']
    assert sum(s.get('rejected_insufficient_capital',0) for s in result.portfolio_signal_stats)==1
    assert sum(s.get('rejected_cooldown',0) for s in result.signal_stats)==1
    assert sum(s.get('rejected_cooldown',0) for s in result.portfolio_signal_stats)==0
    assert (result.equity.buying_power_usd>=0).all()
    assert result.equity.iloc[-1].equity_usd==pytest.approx(100+sum(t['cash_pnl_usd'] for t in result.portfolio_trades))
    assert result.metrics['portfolio_trade_count']==3
    # Overlapping long/short marks are not a sequential sum; short proceeds cannot finance longs.
    flat=market([[10,11,9,11],[11,12,10,12],[12,13,11,12]])
    params=replace(params,initial_capital=100,maximum_position_notional=50,max_signals_per_side=None,
                   cooldown_minutes=0,sl=90,tp=90,holding_minutes=30,commission=.02)
    signals=[dict(datetime=start,long=['L'],short=['S'])]
    with patch('hypertrade.simulation.backtest.ensure_market_data',return_value=flat):
        overlap=evaluate(signals,params)
    quantities=[t['quantity'] for t in overlap.portfolio_trades]
    assert quantities==[5,4]  # entry fees prevent 10 fully-funded shares
    middle=overlap.equity[overlap.equity.datetime==start+pd.Timedelta(minutes=15)].iloc[0]
    assert middle.unrealized_pnl_usd==pytest.approx((11-10)*(5-4)-.02*9)
    assert middle.locked_short_proceeds_usd==40
    assert middle.short_liability_usd==44
    assert middle.equity_usd==pytest.approx(middle.available_cash_usd+middle.long_security_value_usd+2*middle.locked_short_proceeds_usd-middle.short_liability_usd)
    equity=overlap.equity.equity_usd.to_numpy()
    expected=((np.maximum.accumulate(equity)-equity)/np.maximum.accumulate(equity)*100).max()
    assert overlap.metrics['portfolio_max_drawdown_pct']==pytest.approx(expected)
    assert overlap.metrics['portfolio_pnl_usd']==pytest.approx(sum(t['cash_pnl_usd'] for t in overlap.portfolio_trades))
    assert overlap.sequential.iloc[-1].cumulative_normalized_pnl!=pytest.approx(overlap.metrics['portfolio_pnl_usd'])
    # A zero-hold open exit charges both fees and leaves no unsettled position.
    with patch('hypertrade.simulation.backtest.ensure_market_data',return_value=flat):
        zero=evaluate(signals,replace(params,holding_minutes=0))
    assert zero.equity.iloc[-1].reserved_notional_usd==0
    assert zero.metrics['portfolio_pnl_usd']==pytest.approx(-.04*9)
    with patch('hypertrade.simulation.backtest.ensure_market_data',return_value=flat):
        slipped=evaluate([dict(datetime=start,long=['L'],short=[])],replace(params,slippage=.001))
    entry=slipped.equity[slipped.equity.event_phase=='after_exits_then_entries'].iloc[0]
    trade=slipped.portfolio_trades[0]
    assert entry.equity_usd==pytest.approx(100-trade['entry_fee_usd']+(10-trade['entry_price'])*trade['quantity'])
    # A failed indicator cannot consume the cap or alter cooldown for a later eligible signal.
    frames['A']['ema_10']=[0,2,2];frames['A']['ema_40']=[1,1,1]
    frames['C']['ema_10']=[2,2,2];frames['C']['ema_40']=[1,1,1]
    params=replace(params,max_signals_per_side=1,cooldown_minutes=30)
    params.indicator_config['ema'].update(enabled=True,sign='above',fast=10,slow=40)
    signals=[dict(datetime=start,long=['A','C'],short=[]),dict(datetime=start+pd.Timedelta(minutes=15),long=['A'],short=[])]
    with patch('hypertrade.simulation.backtest.ensure_market_data',side_effect=lambda symbol,*args:frames[symbol]):
        eligible=evaluate(signals,params)
    assert [t['symbol'] for t in eligible.trades]==['C','A']
    assert sum(s.get('rejected_indicators_filter_failed',0) for s in eligible.signal_stats)==1
