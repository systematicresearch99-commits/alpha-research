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
from backtests.engine   import run_backtest

# ── Strategy imports ───────────────────────────────────────────────────────────

# Trend Following
from strategies.sma_crossover      import generate_signals as sma_signals,      get_params as sma_params,      STRATEGY_NAME as SMA_NAME
from strategies.ema_crossover      import generate_signals as ema_signals,      get_params as ema_params,      STRATEGY_NAME as EMA_NAME
from strategies.macd               import generate_signals as macd_signals,     get_params as macd_params,     STRATEGY_NAME as MACD_NAME
from strategies.donchian_breakout  import generate_signals as donchian_signals, get_params as donchian_params, STRATEGY_NAME as DONCHIAN_NAME

# Mean Reversion
from strategies.rsi_mean_reversion import generate_signals as rsi_signals,      get_params as rsi_params,      STRATEGY_NAME as RSI_NAME
from strategies.bollinger_reversion import generate_signals as boll_signals,    get_params as boll_params,     STRATEGY_NAME as BOLL_NAME
from strategies.zscore_reversion   import generate_signals as zscore_signals,   get_params as zscore_params,   STRATEGY_NAME as ZSCORE_NAME
from strategies.stochastic         import generate_signals as stoch_signals,    get_params as stoch_params,    STRATEGY_NAME as STOCH_NAME

# Momentum
from strategies.roc_momentum       import generate_signals as roc_signals,      get_params as roc_params,      STRATEGY_NAME as ROC_NAME
from strategies.dual_momentum      import generate_signals as dual_signals,     get_params as dual_params,     STRATEGY_NAME as DUAL_NAME

# Volatility
from strategies.atr_breakout       import generate_signals as atr_signals,      get_params as atr_params,      STRATEGY_NAME as ATR_NAME
from strategies.keltner_breakout   import generate_signals as keltner_signals,  get_params as keltner_params,  STRATEGY_NAME as KELTNER_NAME

# Volume
from strategies.obv_trend          import generate_signals as obv_signals,      get_params as obv_params,      STRATEGY_NAME as OBV_NAME
from strategies.vwap_reversion     import generate_signals as vwap_signals,     get_params as vwap_params,     STRATEGY_NAME as VWAP_NAME

# Statistical
from strategies.kalman_mispricing  import generate_signals as kalman_signals,   get_params as kalman_params,   STRATEGY_NAME as KALMAN_NAME

# ── Strategy registry ──────────────────────────────────────────────────────────
STRATEGIES = {
    # Trend Following
    "sma":          (sma_signals,      sma_params,      SMA_NAME),
    "ema":          (ema_signals,      ema_params,      EMA_NAME),
    "macd":         (macd_signals,     macd_params,     MACD_NAME),
    "donchian":     (donchian_signals, donchian_params, DONCHIAN_NAME),
    # Mean Reversion
    "rsi":          (rsi_signals,      rsi_params,      RSI_NAME),
    "bollinger":    (boll_signals,     boll_params,     BOLL_NAME),
    "zscore":       (zscore_signals,   zscore_params,   ZSCORE_NAME),
    "stochastic":   (stoch_signals,    stoch_params,    STOCH_NAME),
    # Momentum
    "roc":          (roc_signals,      roc_params,      ROC_NAME),
    "dual_momentum":(dual_signals,     dual_params,     DUAL_NAME),
    # Volatility
    "atr":          (atr_signals,      atr_params,      ATR_NAME),
    "keltner":      (keltner_signals,  keltner_params,  KELTNER_NAME),
    # Volume
    "obv":          (obv_signals,      obv_params,      OBV_NAME),
    "vwap":         (vwap_signals,     vwap_params,     VWAP_NAME),
    # Statistical
    "kalman":       (kalman_signals,   kalman_params,   KALMAN_NAME),
    # oil_shock handled separately — two-ticker pipeline
}
# ──────────────────────────────────────────────────────────────────────────────


def run_strategy(strategy_key, ticker, start, end=None,
                 source="yfinance", save=True, notes=None, **strategy_kwargs):
    """
    Full pipeline: load → signal → backtest → metrics → store → print.

    Always loads full OHLCV so every strategy has access to
    High, Low, Volume regardless of what it needs.

    For oil_shock strategy use run_oil_shock() instead.
    """
    if strategy_key not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy_key}'. Choose from: {list(STRATEGIES)}")

    gen_signals, get_p, strat_name = STRATEGIES[strategy_key]

    print(f"\n[run] Loading {ticker} from {source} (start={start})")
    # Always ohlcv=True — all strategies get full OHLCV data
    data = load_data(ticker, start=start, end=end, source=source, ohlcv=True)
    print(f"[run] {len(data)} rows loaded  ({data.index[0].date()} → {data.index[-1].date()})")

    print(f"[run] Generating signals: {strat_name}")
    df = gen_signals(data, **strategy_kwargs)

    print(f"[run] Running backtest...")
    df = run_backtest(df)

    metrics = calculate_metrics(df)
    params  = get_p(**strategy_kwargs)

    print_summary(metrics, strategy_name=f"{strat_name}  [{ticker}]")

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


def run_oil_shock(start="2000-01-01", end=None, shock_col="daily_shock",
                  hold_days=3, notes=None):
    """
    Two-ticker pipeline for Oil Shock Short strategy.
    Results are NOT saved to SQLite — experimental research run.
    """
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

    print(f"[run] {len(df)} rows loaded  ({df.index[0].date()} → {df.index[-1].date()})")

    df["daily_shock_raw"]  = (df["wti_ret"] > DAILY_THRESH).astype(int)
    df["weekly_ret_wti"]   = df["wti_ret"].rolling(5).sum()
    df["weekly_shock_raw"] = (df["weekly_ret_wti"] > WEEKLY_THRESH).astype(int)

    for raw_col, out_col in [("daily_shock_raw", "daily_shock"),
                              ("weekly_shock_raw", "weekly_shock")]:
        flags   = df[raw_col].values.copy()
        cleaned = np.zeros(len(flags), dtype=int)
        last    = -EXCLUSION_DAYS - 1
        for i in range(len(flags)):
            if flags[i] == 1 and (i - last) > EXCLUSION_DAYS:
                cleaned[i] = 1
                last = i
        df[out_col] = cleaned

    n_shocks = df[shock_col].sum()
    print(f"[run] {shock_col}: {n_shocks} events identified")

    print(f"[run] Generating signals: {OIL_NAME}")
    df = oil_signals(df, shock_col=shock_col, hold_days=hold_days)

    print(f"[run] Running backtest...")
    df = run_backtest(df)

    metrics = calculate_metrics(df)
    params  = oil_params(shock_col=shock_col, hold_days=hold_days)

    print_summary(metrics, strategy_name=f"{OIL_NAME}  [^GSPC | {shock_col} | hold={hold_days}d]")

    if notes:
        print(f"[run] Notes: {notes}")
    print(f"[run] Oil shock run complete (not saved to research log — experimental)")

    return df, metrics


def main():
    parser = argparse.ArgumentParser(
        description="AlphaByProcess — Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy keys:
  Trend:      sma, ema, macd, donchian
  Reversion:  rsi, bollinger, zscore, stochastic
  Momentum:   roc, dual_momentum
  Volatility: atr, keltner
  Volume:     obv, vwap
  Statistical:kalman
  Macro:      oil_shock
        """
    )
    parser.add_argument("--strategy",   default="sma",         help="Strategy key (see list above)")
    parser.add_argument("--ticker",     default="BTC-USD",     help="Ticker symbol (not used for oil_shock)")
    parser.add_argument("--start",      default="2020-01-01",  help="Start date")
    parser.add_argument("--end",        default=None,          help="End date (optional)")
    parser.add_argument("--source",     default="yfinance",    help="Data source: yfinance | binance | csv")
    parser.add_argument("--compare",    action="store_true",   help="Print comparison of all saved runs")
    parser.add_argument("--no-save",    action="store_true",   help="Don't save results to DB")
    parser.add_argument("--notes",      default=None,          help="Research note to attach")

    # Oil shock specific
    parser.add_argument("--shock-col",  default="daily_shock", help="daily_shock | weekly_shock")
    parser.add_argument("--hold-days",  default=3, type=int,   help="Days to hold short (oil_shock only)")

    # Kalman specific
    parser.add_argument("--obs-noise",  default=1.0,  type=float, help="Kalman observation noise R (default 1.0)")
    parser.add_argument("--proc-noise", default=0.01, type=float, help="Kalman process noise Q (default 0.01)")
    parser.add_argument("--entry-z",    default=1.5,  type=float, help="Kalman entry z-score threshold (default 1.5)")
    parser.add_argument("--exit-z",     default=0.3,  type=float, help="Kalman exit z-score threshold (default 0.3)")
    parser.add_argument("--stop-z",     default=3.5,  type=float, help="Kalman stop-loss z-score (default 3.5)")

    args = parser.parse_args()

    if args.compare:
        compare_strategies()
        return

    if args.strategy == "oil_shock":
        run_oil_shock(
            start     = args.start,
            end       = args.end,
            shock_col = args.shock_col,
            hold_days = args.hold_days,
            notes     = args.notes,
        )
    elif args.strategy == "kalman":
        run_strategy(
            strategy_key   = "kalman",
            ticker         = args.ticker,
            start          = args.start,
            end            = args.end,
            source         = args.source,
            save           = not args.no_save,
            notes          = args.notes,
            obs_noise_var  = args.obs_noise,
            proc_noise_var = args.proc_noise,
            entry_z        = args.entry_z,
            exit_z         = args.exit_z,
            stop_loss_z    = args.stop_z,
        )
    else:
        run_strategy(
            strategy_key = args.strategy,
            ticker       = args.ticker,
            start        = args.start,
            end          = args.end,
            source       = args.source,
            save         = not args.no_save,
            notes        = args.notes,
        )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("  AlphaByProcess — Running demo: BTC-USD 2020→2024")
        print("=" * 60)

        # Trend Following
        run_strategy("sma",      "BTC-USD", start="2020-01-01", notes="Baseline SMA")
        run_strategy("ema",      "BTC-USD", start="2020-01-01", notes="Baseline EMA")
        run_strategy("macd",     "BTC-USD", start="2020-01-01", notes="Baseline MACD")
        run_strategy("donchian", "BTC-USD", start="2020-01-01", notes="Baseline Donchian")

        # Mean Reversion
        run_strategy("rsi",        "BTC-USD", start="2020-01-01", notes="Baseline RSI")
        run_strategy("bollinger",  "BTC-USD", start="2020-01-01", notes="Baseline Bollinger")
        run_strategy("zscore",     "BTC-USD", start="2020-01-01", notes="Baseline ZScore")
        run_strategy("stochastic", "BTC-USD", start="2020-01-01", notes="Baseline Stochastic")

        # Momentum
        run_strategy("roc",          "BTC-USD", start="2020-01-01", notes="Baseline ROC")
        run_strategy("dual_momentum","BTC-USD", start="2020-01-01", notes="Baseline Dual Momentum")

        # Volatility
        run_strategy("atr",     "BTC-USD", start="2020-01-01", notes="Baseline ATR Breakout")
        run_strategy("keltner", "BTC-USD", start="2020-01-01", notes="Baseline Keltner")

        # Volume
        run_strategy("obv",  "BTC-USD", start="2020-01-01", notes="Baseline OBV")
        run_strategy("vwap", "BTC-USD", start="2020-01-01", notes="Baseline VWAP")

        # Statistical
        run_strategy("kalman", "BTC-USD", start="2020-01-01", notes="Baseline Kalman")

        print("\n── Comparison of all saved runs ──")
        compare_strategies()
    else:
        main()