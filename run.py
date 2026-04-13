"""
run.py — Main research orchestrator for AlphaByProcess framework.

Usage:
    # Run a single strategy
    python run.py

    # Compare all saved runs
    python run.py --compare

    # Run with custom params
    python run.py --strategy rsi --ticker BTC-USD --start 2020-01-01

    # Run Kalman mispricing strategy
    python run.py --strategy kalman --ticker SPY --start 2018-01-01
    python run.py --strategy kalman --ticker SPY --entry-z 1.2 --exit-z 0.2 --obs-noise 2.0 --proc-noise 0.05

    # Run oil shock strategy (two-ticker pipeline)
    python run.py --strategy oil_shock --start 2000-01-01

Available strategies
--------------------
Trend Following:
    sma             SMA Crossover          (short_window, long_window)
    ema             EMA Crossover          (short_window, long_window)
    macd            MACD                   (fast, slow, signal_period)
    donchian        Donchian Breakout      (window)

Mean Reversion:
    rsi             RSI Mean Reversion     (rsi_period, oversold, overbought)
    bollinger       Bollinger Reversion    (window, num_std)
    zscore          Z-Score Reversion      (window, entry_z, exit_z)
    stochastic      Stochastic Oscillator  (k_period, d_period, oversold, overbought)

Momentum:
    roc             ROC Momentum           (window, threshold)
    dual_momentum   Dual Momentum          (lookback, risk_free_rate)

Volatility:
    atr             ATR Breakout           (atr_period, multiplier)
    keltner         Keltner Breakout       (ema_period, atr_period, multiplier)

Volume:
    obv             OBV Trend              (obv_ma_period)
    vwap            VWAP Reversion         (window, entry_pct, exit_pct)

Statistical:
    kalman          Kalman Mispricing      (obs_noise_var, proc_noise_var, entry_z, exit_z, stop_loss_z)

Event / Macro:
    oil_shock       Oil Shock Short        (shock_col, hold_days)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader  import load_data
from utils.performance  import calculate_metrics, print_summary, _extract_trades
from utils.store        import save_run, compare_strategies
from utils.plotting     import plot_run, plot_overlay, block
from backtests.engine   import run_backtest

from strategies.sma_crossover       import generate_signals as sma_signals,      get_params as sma_params,      STRATEGY_NAME as SMA_NAME
from strategies.ema_crossover       import generate_signals as ema_signals,      get_params as ema_params,      STRATEGY_NAME as EMA_NAME
from strategies.macd                import generate_signals as macd_signals,     get_params as macd_params,     STRATEGY_NAME as MACD_NAME
from strategies.donchian_breakout   import generate_signals as donchian_signals, get_params as donchian_params, STRATEGY_NAME as DONCHIAN_NAME
from strategies.rsi_mean_reversion  import generate_signals as rsi_signals,      get_params as rsi_params,      STRATEGY_NAME as RSI_NAME
from strategies.bollinger_reversion import generate_signals as boll_signals,     get_params as boll_params,     STRATEGY_NAME as BOLL_NAME
from strategies.zscore_reversion    import generate_signals as zscore_signals,   get_params as zscore_params,   STRATEGY_NAME as ZSCORE_NAME
from strategies.stochastic          import generate_signals as stoch_signals,    get_params as stoch_params,    STRATEGY_NAME as STOCH_NAME
from strategies.roc_momentum        import generate_signals as roc_signals,      get_params as roc_params,      STRATEGY_NAME as ROC_NAME
from strategies.dual_momentum       import generate_signals as dual_signals,     get_params as dual_params,     STRATEGY_NAME as DUAL_NAME
from strategies.atr_breakout        import generate_signals as atr_signals,      get_params as atr_params,      STRATEGY_NAME as ATR_NAME
from strategies.keltner_breakout    import generate_signals as keltner_signals,  get_params as keltner_params,  STRATEGY_NAME as KELTNER_NAME
from strategies.obv_trend           import generate_signals as obv_signals,      get_params as obv_params,      STRATEGY_NAME as OBV_NAME
from strategies.vwap_reversion      import generate_signals as vwap_signals,     get_params as vwap_params,     STRATEGY_NAME as VWAP_NAME
from strategies.kalman_mispricing   import generate_signals as kalman_signals,   get_params as kalman_params,   STRATEGY_NAME as KALMAN_NAME

STRATEGIES = {
    "sma":           (sma_signals,      sma_params,      SMA_NAME),
    "ema":           (ema_signals,      ema_params,      EMA_NAME),
    "macd":          (macd_signals,     macd_params,     MACD_NAME),
    "donchian":      (donchian_signals, donchian_params, DONCHIAN_NAME),
    "rsi":           (rsi_signals,      rsi_params,      RSI_NAME),
    "bollinger":     (boll_signals,     boll_params,     BOLL_NAME),
    "zscore":        (zscore_signals,   zscore_params,   ZSCORE_NAME),
    "stochastic":    (stoch_signals,    stoch_params,    STOCH_NAME),
    "roc":           (roc_signals,      roc_params,      ROC_NAME),
    "dual_momentum": (dual_signals,     dual_params,     DUAL_NAME),
    "atr":           (atr_signals,      atr_params,      ATR_NAME),
    "keltner":       (keltner_signals,  keltner_params,  KELTNER_NAME),
    "obv":           (obv_signals,      obv_params,      OBV_NAME),
    "vwap":          (vwap_signals,     vwap_params,     VWAP_NAME),
    "kalman":        (kalman_signals,   kalman_params,   KALMAN_NAME),
}


# ── Core single-run pipeline ──────────────────────────────────────────────────

def run_strategy(strategy_key, ticker, start, end=None,
                 source="yfinance", save=True, notes=None,
                 show_chart=True, **strategy_kwargs):
    """
    Single strategy × single ticker pipeline.
    Returns (df, metrics). Chart is non-blocking if show_chart=True.
    """
    if strategy_key not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy_key}'. Choose from: {list(STRATEGIES)}")

    gen_signals, get_p, strat_name = STRATEGIES[strategy_key]

    print(f"\n[run] Loading {ticker} from {source} (start={start})")
    data = load_data(ticker, start=start, end=end, source=source, ohlcv=True)
    print(f"[run] {len(data)} rows loaded  ({data.index[0].date()} → {data.index[-1].date()})")

    print(f"[run] Generating signals: {strat_name}")
    df = gen_signals(data, **strategy_kwargs)

    print(f"[run] Running backtest...")
    df = run_backtest(df)

    metrics = calculate_metrics(df)
    params  = get_p(**strategy_kwargs)

    print_summary(metrics, strategy_name=f"{strat_name}  [{ticker}]")

    if show_chart:
        plot_run(df, title=f"{strat_name}  |  {ticker}  |  {data.index[0].date()} → {data.index[-1].date()}")

    if save:
        trades_df = _extract_trades(df)
        save_run(
            strategy   = strat_name,
            ticker     = ticker,
            metrics    = metrics,
            params     = params,
            start_date = str(data.index[0].date()),
            end_date   = str(data.index[-1].date()),
            trades_df  = trades_df,
            notes      = notes,
        )

    return df, metrics


# ── Multi-run pipelines ───────────────────────────────────────────────────────

def run_multi_strategy(strategy_keys, ticker, start, end=None,
                       source="yfinance", save=True, notes=None,
                       overlay=False, **strategy_kwargs):
    """
    Multiple strategies × one ticker.

    overlay=False : separate chart per strategy
    overlay=True  : one overlaid chart with all equity curves + summary table
    """
    results = {}
    print(f"\n[run] Multi-strategy run — {len(strategy_keys)} strategies on {ticker}")
    print(f"[run] Strategies: {', '.join(strategy_keys)}")

    for key in strategy_keys:
        df, metrics = run_strategy(
            key, ticker, start, end=end, source=source,
            save=save, notes=notes,
            show_chart=not overlay,   # suppress individual charts if overlaying
            **strategy_kwargs,
        )
        label = f"{STRATEGIES[key][2]}"
        results[label] = df

    if overlay and results:
        plot_overlay(results, title=f"Strategy Comparison  |  {ticker}  |  {start} → {end or 'today'}")

    return results


def run_multi_ticker(strategy_key, tickers, start, end=None,
                     source="yfinance", save=True, notes=None,
                     overlay=False, **strategy_kwargs):
    """
    One strategy × multiple tickers.

    overlay=False : separate chart per ticker
    overlay=True  : one overlaid chart with all equity curves + summary table
    """
    gen_signals, get_p, strat_name = STRATEGIES[strategy_key]
    results = {}
    print(f"\n[run] Multi-ticker run — {strat_name} on {len(tickers)} tickers")
    print(f"[run] Tickers: {', '.join(tickers)}")

    for ticker in tickers:
        df, metrics = run_strategy(
            strategy_key, ticker, start, end=end, source=source,
            save=save, notes=notes,
            show_chart=not overlay,
            **strategy_kwargs,
        )
        results[ticker] = df

    if overlay and results:
        plot_overlay(results, title=f"{strat_name}  |  Ticker Comparison  |  {start} → {end or 'today'}")

    return results


def run_matrix(strategy_keys, tickers, start, end=None,
               source="yfinance", save=True, notes=None,
               overlay=False, **strategy_kwargs):
    """
    Multiple strategies × multiple tickers.

    Runs every combination. When overlay=True, produces one overlay
    chart per ticker with all strategies on it.
    """
    print(f"\n[run] Matrix run — {len(strategy_keys)} strategies × {len(tickers)} tickers")
    print(f"[run] Strategies : {', '.join(strategy_keys)}")
    print(f"[run] Tickers    : {', '.join(tickers)}")

    all_results = {}

    for ticker in tickers:
        ticker_results = {}
        for key in strategy_keys:
            df, metrics = run_strategy(
                key, ticker, start, end=end, source=source,
                save=save, notes=notes,
                show_chart=not overlay,
                **strategy_kwargs,
            )
            label = STRATEGIES[key][2]
            ticker_results[label] = df
            all_results[f"{label} | {ticker}"] = df

        if overlay and ticker_results:
            plot_overlay(
                ticker_results,
                title=f"Strategy Comparison  |  {ticker}  |  {start} → {end or 'today'}"
            )

    return all_results


def run_oil_shock(start="2000-01-01", end=None, shock_col="daily_shock",
                  hold_days=3, notes=None):
    """Two-ticker pipeline for Oil Shock Short strategy."""
    import numpy as np
    from strategies.oil_shock_short import generate_signals as oil_signals, STRATEGY_NAME as OIL_NAME
    from strategies.oil_shock_short import get_params as oil_params

    DAILY_THRESH   = 0.05
    WEEKLY_THRESH  = 0.10
    EXCLUSION_DAYS = 5

    print(f"\n[run] Oil Shock pipeline — loading WTI (CL=F)...")
    wti = load_data("CL=F", start=start, end=end, source="yfinance", ohlcv=False)
    wti.columns = ["WTI"]

    print(f"[run] Loading S&P 500 (^GSPC)...")
    sp500 = load_data("^GSPC", start=start, end=end, source="yfinance", ohlcv=True)

    df = sp500.join(wti, how="inner")
    df["wti_ret"] = df["WTI"].pct_change()
    df.dropna(inplace=True)

    df["daily_shock_raw"]  = (df["wti_ret"] > DAILY_THRESH).astype(int)
    df["weekly_ret_wti"]   = df["wti_ret"].rolling(5).sum()
    df["weekly_shock_raw"] = (df["weekly_ret_wti"] > WEEKLY_THRESH).astype(int)

    for raw_col, out_col in [("daily_shock_raw", "daily_shock"),
                              ("weekly_shock_raw", "weekly_shock")]:
        flags   = df[raw_col].values.copy()
        cleaned = __import__("numpy").zeros(len(flags), dtype=int)
        last    = -EXCLUSION_DAYS - 1
        for i in range(len(flags)):
            if flags[i] == 1 and (i - last) > EXCLUSION_DAYS:
                cleaned[i] = 1
                last = i
        df[out_col] = cleaned

    print(f"[run] {shock_col}: {df[shock_col].sum()} events identified")

    df = oil_signals(df, shock_col=shock_col, hold_days=hold_days)
    df = run_backtest(df)

    metrics = calculate_metrics(df)
    print_summary(metrics, strategy_name=f"{OIL_NAME}  [^GSPC | {shock_col} | hold={hold_days}d]")
    plot_run(df, title=f"{OIL_NAME}  |  ^GSPC  |  {shock_col}  |  hold={hold_days}d")

    return df, metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AlphaByProcess — Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --strategy macd --ticker HDFCBANK.NS --start 2018-01-01
  python run.py --strategies sma ema macd --ticker HDFCBANK.NS
  python run.py --strategies sma ema macd --ticker HDFCBANK.NS --overlay
  python run.py --strategy macd --tickers HDFCBANK.NS RELIANCE.NS TCS.NS
  python run.py --strategies sma macd rsi --tickers HDFCBANK.NS RELIANCE.NS --overlay
        """
    )

    # Strategy — singular or plural
    strat_group = parser.add_mutually_exclusive_group()
    strat_group.add_argument("--strategy",   default=None,        help="Single strategy key")
    strat_group.add_argument("--strategies", nargs="+",           help="Multiple strategy keys")

    # Ticker — singular or plural
    ticker_group = parser.add_mutually_exclusive_group()
    ticker_group.add_argument("--ticker",    default=None,        help="Single ticker")
    ticker_group.add_argument("--tickers",   nargs="+",           help="Multiple tickers")

    parser.add_argument("--start",      default="2015-01-01",     help="Start date")
    parser.add_argument("--end",        default=None,             help="End date")
    parser.add_argument("--source",     default="yfinance",       help="Data source")
    parser.add_argument("--overlay",    action="store_true",      help="Overlay equity curves on one chart")
    parser.add_argument("--compare",    action="store_true",      help="Print comparison of all saved runs")
    parser.add_argument("--no-save",    action="store_true",      help="Don't save to DB")
    parser.add_argument("--notes",      default=None)

    parser.add_argument("--shock-col",  default="daily_shock")
    parser.add_argument("--hold-days",  default=3,    type=int)
    parser.add_argument("--obs-noise",  default=1.0,  type=float)
    parser.add_argument("--proc-noise", default=0.01, type=float)
    parser.add_argument("--entry-z",    default=1.5,  type=float)
    parser.add_argument("--exit-z",     default=0.3,  type=float)
    parser.add_argument("--stop-z",     default=3.5,  type=float)

    args = parser.parse_args()

    if args.compare:
        compare_strategies()
        return

    # Resolve strategy list and ticker list
    strategies = args.strategies or ([args.strategy] if args.strategy else ["sma"])
    tickers    = args.tickers    or ([args.ticker]    if args.ticker    else ["SPY"])

    # Oil shock — special case, ignore multi flags
    if strategies == ["oil_shock"]:
        run_oil_shock(start=args.start, end=args.end,
                      shock_col=args.shock_col, hold_days=args.hold_days, notes=args.notes)
        block()
        return

    # Kalman kwargs
    strategy_kwargs = {}
    if "kalman" in strategies:
        strategy_kwargs = dict(
            obs_noise_var=args.obs_noise, proc_noise_var=args.proc_noise,
            entry_z=args.entry_z, exit_z=args.exit_z, stop_loss_z=args.stop_z,
        )

    save = not args.no_save

    # Dispatch to correct pipeline
    if len(strategies) == 1 and len(tickers) == 1:
        run_strategy(strategies[0], tickers[0], args.start, args.end,
                     args.source, save, args.notes, **strategy_kwargs)

    elif len(strategies) > 1 and len(tickers) == 1:
        run_multi_strategy(strategies, tickers[0], args.start, args.end,
                           args.source, save, args.notes,
                           overlay=args.overlay, **strategy_kwargs)

    elif len(strategies) == 1 and len(tickers) > 1:
        run_multi_ticker(strategies[0], tickers, args.start, args.end,
                         args.source, save, args.notes,
                         overlay=args.overlay, **strategy_kwargs)

    else:
        run_matrix(strategies, tickers, args.start, args.end,
                   args.source, save, args.notes,
                   overlay=args.overlay, **strategy_kwargs)

    block()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("  AlphaByProcess — Demo: BTC-USD 2020→2024")
        print("=" * 60)

        run_multi_strategy(
            list(STRATEGIES.keys()), "BTC-USD",
            start="2020-01-01", overlay=True,
        )
        compare_strategies()
        block()
    else:
        main()