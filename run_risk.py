"""
run_risk.py — Risk Module Research Runner
==========================================
Extends the AlphaByProcess framework with position sizing and
stop loss research. Mirrors run.py but adds a risk layer between
signal generation and performance measurement.

Usage
-----
    # Single risk module
    python run_risk.py --strategy macd --ticker SPY --start 2015-01-01 --risk atr_sizing

    # Stack multiple modules (one combined backtest)
    python run_risk.py --strategy rsi --ticker BTC-USD --risk atr_sizing atr_trailing_stop

    # Compare multiple modules side by side
    python run_risk.py --strategy sma --ticker SPY --risk fixed_fractional kelly vol_target --compare

    # With regime labels attached (enables regime_sizer)
    python run_risk.py --strategy macd --ticker SPY --risk regime_sizer --regime

    # List all available risk modules
    python run_risk.py --list-modules

    # No save — research only
    python run_risk.py --strategy rsi --ticker BTC-USD --risk kelly --no-save

Available risk modules
----------------------
Sizing:
    fixed_fractional   Risk fixed % of capital per trade
    atr_sizing         Size so 1 ATR move = fixed % of capital
    kelly              Optimal fraction from rolling win rate + win/loss ratio
    vol_target         Scale size to hit constant realized volatility
    drawdown_adaptive  Shrink size as equity drawdown deepens
    regime_sizer       Scale size by HMM regime (requires --regime)
    signal_strength    Size proportional to indicator signal intensity

Stops:
    fixed_stop         Exit if price drops N% from entry
    atr_trailing_stop  Trail stop by N * ATR, ratchets with price
    time_stop          Exit after N bars regardless of P&L

Available strategies
--------------------
Trend:      sma, ema, macd, donchian
Reversion:  rsi, bollinger, zscore, stochastic
Momentum:   roc, dual_momentum
Volatility: atr, keltner
Volume:     obv, vwap
Statistical:kalman
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader  import load_data
from utils.performance  import calculate_metrics, print_summary, _extract_trades
from utils.store        import save_run, compare_strategies
from backtests.engine   import run_backtest
from risk.risk_manager  import apply_stack, apply_compare, list_modules, REGISTRY

# ── Strategy imports ───────────────────────────────────────────────────────────
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _attach_regime(df, data, index_ticker, start, end, feature_window=20):
    """Train HMM and attach regime labels to df."""
    from research.regime_detection.hmm_model       import RegimeDetector
    from research.regime_detection.regime_analyzer import RegimeAnalyzer

    print(f"  [risk] Loading benchmark index: {index_ticker}")
    index_data = load_data(index_ticker, start=start, end=end, source="yfinance", ohlcv=True)

    print(f"  [risk] Training HMM regime detector...")
    detector = RegimeDetector(n_regimes=4, n_iter=1000, window=feature_window)
    detector.fit(data["Close"], index_data["Close"])

    analyzer = RegimeAnalyzer(detector, index_data["Close"])
    df       = analyzer.attach_regimes(df, data["Close"])
    print(f"  [risk] Regime labels attached.")
    return df


def _print_risk_summary(metrics, label):
    """Print a compact one-line risk result."""
    sr   = metrics.get("Sharpe Ratio", float("nan"))
    tr   = metrics.get("Total Return", float("nan"))
    mdd  = metrics.get("Max Drawdown", float("nan"))
    wr   = metrics.get("Win Rate", float("nan"))
    print(f"  {label:<30}  Sharpe={sr:>6.3f}  Return={tr*100:>7.2f}%  "
          f"MaxDD={mdd*100:>7.2f}%  WinRate={wr*100:>5.1f}%")


def _save_risk_run(strategy_name, ticker, metrics, params, risk_keys,
                   start_date, end_date, df, notes, db_path=None):
    trades_df = _extract_trades(df)
    run_id = save_run(
        strategy   = f"{strategy_name}+{'|'.join(risk_keys)}",
        ticker     = ticker,
        metrics    = metrics,
        params     = params,
        start_date = start_date,
        end_date   = end_date,
        trades_df  = trades_df,
        notes      = notes,
        db_path    = db_path,
    )
    return run_id


# ── Core pipeline ──────────────────────────────────────────────────────────────

def run_risk(
    strategy_key,
    ticker,
    start,
    risk_modules,
    end             = None,
    source          = "yfinance",
    compare         = False,
    use_regime      = False,
    index_ticker    = "SPY",
    feature_window  = 20,
    save            = True,
    notes           = None,
    strategy_kwargs = None,
    module_kwargs   = None,
):
    """
    Full pipeline:
      load → signal → base backtest → risk modules → metrics → store → print

    Args:
        strategy_key    : strategy key from STRATEGIES dict
        ticker          : ticker symbol e.g. "SPY", "BTC-USD"
        start           : start date string
        risk_modules    : list of risk module keys to apply
        end             : end date string (optional)
        source          : "yfinance" | "binance" | "csv"
        compare         : if True, run modules side by side instead of stacking
        use_regime      : if True, train HMM and attach regime labels
        index_ticker    : benchmark for HMM (default "SPY")
        feature_window  : HMM feature window (default 20)
        save            : persist to SQLite (default True)
        notes           : research note string
        strategy_kwargs : dict of strategy params e.g. {"short_window": 10}
        module_kwargs   : dict of {module_key: {param: value}}

    Returns:
        (base_df, risk_result)
        risk_result is a dict {module_key: df} if compare=True,
        or a single df if stacking.
    """
    if strategy_key not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy_key}'. Choose from: {list(STRATEGIES)}")

    invalid = [k for k in risk_modules if k not in REGISTRY]
    if invalid:
        raise ValueError(f"Unknown risk module(s): {invalid}. Choose from: {list(REGISTRY)}")

    strategy_kwargs = strategy_kwargs or {}
    module_kwargs   = module_kwargs   or {}

    gen_signals, get_p, strat_name = STRATEGIES[strategy_key]

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n[risk] Loading {ticker} from {source} (start={start})")
    data = load_data(ticker, start=start, end=end, source=source, ohlcv=True)
    print(f"[risk] {len(data)} rows loaded  ({data.index[0].date()} → {data.index[-1].date()})")

    # ── 2. Signal ─────────────────────────────────────────────────────────────
    print(f"[risk] Generating signals: {strat_name}")
    df = gen_signals(data, **strategy_kwargs)

    # ── 3. Base backtest (needed for drawdown_adaptive and baseline comparison)
    print(f"[risk] Running base backtest...")
    df = run_backtest(df)

    base_metrics = calculate_metrics(df)
    print_summary(base_metrics, strategy_name=f"{strat_name}  [{ticker}]  (no risk)")

    # ── 4. Attach regime labels if needed ─────────────────────────────────────
    if use_regime or "regime_sizer" in risk_modules:
        df = _attach_regime(df, data, index_ticker, start, end, feature_window)

    # ── 5. Apply risk modules ─────────────────────────────────────────────────
    risk_label = " + ".join(risk_modules)

    if compare:
        print(f"\n[risk] Comparing {len(risk_modules)} modules side by side...")
        results = apply_compare(df, risk_modules, module_kwargs)

        print(f"\n{'═'*75}")
        print(f"  Risk Module Comparison  —  {strat_name} [{ticker}]")
        print(f"{'═'*75}")

        # Print baseline first
        print(f"\n  {'Module':<30}  {'Sharpe':>6}  {'Return':>8}  {'MaxDD':>8}  {'WinRate':>8}")
        print(f"  {'─'*70}")
        _print_risk_summary(base_metrics, "Baseline (no risk)")

        all_metrics = {"baseline": (base_metrics, df)}

        for key, risk_df in results.items():
            risk_metrics = calculate_metrics(risk_df)
            all_metrics[key] = (risk_metrics, risk_df)
            _print_risk_summary(risk_metrics, REGISTRY[key][2])

            if save:
                params = {**get_p(**strategy_kwargs), **module_kwargs.get(key, {})}
                _save_risk_run(
                    strat_name, ticker, risk_metrics, params,
                    [key], str(data.index[0].date()), str(data.index[-1].date()),
                    risk_df, notes,
                )

        print(f"{'═'*75}\n")
        return df, results

    else:
        print(f"\n[risk] Stacking: {risk_label}")
        risk_df      = apply_stack(df, risk_modules, module_kwargs)
        risk_metrics = calculate_metrics(risk_df)

        print_summary(risk_metrics, strategy_name=f"{strat_name}  [{ticker}]  ({risk_label})")

        # Side-by-side delta vs baseline
        print(f"\n  Delta vs baseline (no risk):")
        for metric in ["Sharpe Ratio", "Total Return", "Max Drawdown", "Win Rate"]:
            base_val = base_metrics.get(metric, 0) or 0
            risk_val = risk_metrics.get(metric, 0) or 0
            delta    = risk_val - base_val
            sign     = "+" if delta >= 0 else ""
            is_pct   = metric in ["Total Return", "Max Drawdown", "Win Rate"]
            val_str  = f"{sign}{delta*100:.2f}%" if is_pct else f"{sign}{delta:.4f}"
            print(f"    {metric:<22} {val_str}")

        if save:
            params = {**get_p(**strategy_kwargs), **{k: v for d in module_kwargs.values() for k, v in d.items()}}
            run_id = _save_risk_run(
                strat_name, ticker, risk_metrics, params,
                risk_modules, str(data.index[0].date()), str(data.index[-1].date()),
                risk_df, notes,
            )
            print(f"\n[risk] Run saved → ID {run_id}")

        return df, risk_df


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AlphaByProcess — Risk Module Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single module
  python run_risk.py --strategy macd --ticker SPY --start 2015-01-01 --risk atr_sizing

  # Stack (combined backtest)
  python run_risk.py --strategy rsi --ticker BTC-USD --risk atr_sizing atr_trailing_stop

  # Compare side by side
  python run_risk.py --strategy sma --ticker SPY --risk fixed_fractional kelly vol_target --compare

  # With regime labels
  python run_risk.py --strategy macd --ticker SPY --risk regime_sizer --regime
        """
    )

    parser.add_argument("--strategy",   default="sma",        help="Strategy key")
    parser.add_argument("--ticker",     default="SPY",        help="Ticker symbol")
    parser.add_argument("--start",      default="2015-01-01", help="Start date")
    parser.add_argument("--end",        default=None,         help="End date (optional)")
    parser.add_argument("--source",     default="yfinance",   help="Data source")

    parser.add_argument("--risk",       nargs="+",            help="Risk module key(s)", required=False)
    parser.add_argument("--compare",    action="store_true",  help="Compare modules side by side")
    parser.add_argument("--regime",     action="store_true",  help="Attach HMM regime labels")
    parser.add_argument("--index",      default="SPY",        help="Benchmark index for HMM")
    parser.add_argument("--window",     default=20, type=int, help="HMM feature window")

    parser.add_argument("--no-save",    action="store_true",  help="Don't save to DB")
    parser.add_argument("--compare-all",action="store_true",  help="Compare all saved runs in DB")
    parser.add_argument("--list-modules",action="store_true", help="List all available risk modules")
    parser.add_argument("--notes",      default=None,         help="Research note")

    # Kalman-specific
    parser.add_argument("--obs-noise",  default=1.0,  type=float)
    parser.add_argument("--proc-noise", default=0.01, type=float)
    parser.add_argument("--entry-z",    default=1.5,  type=float)
    parser.add_argument("--exit-z",     default=0.3,  type=float)
    parser.add_argument("--stop-z",     default=3.5,  type=float)

    args = parser.parse_args()

    if args.list_modules:
        list_modules()
        return

    if args.compare_all:
        compare_strategies()
        return

    if not args.risk:
        parser.error("--risk is required. Use --list-modules to see options.")

    strategy_kwargs = {}
    if args.strategy == "kalman":
        strategy_kwargs = dict(
            obs_noise_var  = args.obs_noise,
            proc_noise_var = args.proc_noise,
            entry_z        = args.entry_z,
            exit_z         = args.exit_z,
            stop_loss_z    = args.stop_z,
        )

    run_risk(
        strategy_key   = args.strategy,
        ticker         = args.ticker,
        start          = args.start,
        end            = args.end,
        source         = args.source,
        risk_modules   = args.risk,
        compare        = args.compare,
        use_regime     = args.regime,
        index_ticker   = args.index,
        feature_window = args.window,
        save           = not args.no_save,
        notes          = args.notes,
        strategy_kwargs= strategy_kwargs,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 65)
        print("  AlphaByProcess — Risk Module Demo")
        print("  MACD on SPY 2015→2024")
        print("=" * 65)

        # Demo 1: Compare all sizing modules
        run_risk(
            strategy_key = "macd",
            ticker       = "SPY",
            start        = "2015-01-01",
            risk_modules = ["fixed_fractional", "atr_sizing", "kelly", "vol_target"],
            compare      = True,
            notes        = "Sizing module comparison — MACD SPY",
        )

        # Demo 2: Stack sizing + stop
        run_risk(
            strategy_key = "macd",
            ticker       = "SPY",
            start        = "2015-01-01",
            risk_modules = ["atr_sizing", "atr_trailing_stop"],
            compare      = False,
            notes        = "ATR sizing + ATR trailing stop stacked — MACD SPY",
        )

        print("\n── All saved runs ──")
        compare_strategies()
    else:
        main()
