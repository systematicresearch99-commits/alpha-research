"""
run_regime.py — Regime Detection + Strategy Analysis Runner
============================================================
Extends the AlphaByProcess framework with HMM-based market regime
detection and per-regime strategy performance analysis.

Usage
-----
    # Train HMM + analyze one strategy across regimes
    python run_regime.py

    # Specify strategy and ticker
    python run_regime.py --strategy kalman --ticker SPY --start 2018-01-01

    # Compare multiple strategies across regimes (includes Kalman)
    python run_regime.py --compare-strategies

    # Use a previously saved model (skip retraining)
    python run_regime.py --load-model results/regime_model.pkl

    # Save the trained HMM model
    python run_regime.py --save-model results/regime_model.pkl
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader  import load_data
from utils.performance  import calculate_metrics, print_summary, _extract_trades
from utils.store        import save_run, compare_strategies
from backtests.engine   import run_backtest

from strategies.sma_crossover      import generate_signals as sma_signals,    get_params as sma_params,    STRATEGY_NAME as SMA_NAME
from strategies.rsi_mean_reversion import generate_signals as rsi_signals,    get_params as rsi_params,    STRATEGY_NAME as RSI_NAME
from strategies.kalman_mispricing  import generate_signals as kalman_signals, get_params as kalman_params, STRATEGY_NAME as KALMAN_NAME

from research.regime_detection.hmm_model       import RegimeDetector
from research.regime_detection.regime_analyzer import RegimeAnalyzer

# ── Strategy registry ─────────────────────────────────────────────────────────
STRATEGIES = {
    "sma":    (sma_signals,    sma_params,    SMA_NAME),
    "rsi":    (rsi_signals,    rsi_params,    RSI_NAME),
    "kalman": (kalman_signals, kalman_params, KALMAN_NAME),
}

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INDEX  = "SPY"
DEFAULT_MODEL  = "results/regime_model.pkl"
DEFAULT_WINDOW = 20


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_regime_analysis(
    strategy_key,
    ticker,
    start,
    end            = None,
    source         = "yfinance",
    index_ticker   = DEFAULT_INDEX,
    feature_window = DEFAULT_WINDOW,
    save           = True,
    notes          = None,
    model_path     = None,
    save_model     = None,
    **strategy_kwargs,
):
    """
    Full pipeline:
      load → signal → backtest → train HMM → attach regimes → per-regime metrics

    Parameters
    ----------
    strategy_key   : "sma" | "rsi" | "kalman"
    ticker         : e.g. "BTC-USD", "SPY"
    start          : start date string e.g. "2018-01-01"
    end            : end date string (optional, defaults to today)
    source         : "yfinance" | "binance" | "csv"
    index_ticker   : benchmark ticker for index_correlation feature (default "SPY")
    feature_window : rolling window for feature computation (default 20 days)
    save           : persist run to SQLite via store.py (default True)
    notes          : optional research note string
    model_path     : path to load a pre-trained model (skips training if provided)
    save_model     : path to save the trained model (optional)
    **strategy_kwargs : passed through to generate_signals()

    Returns
    -------
    (df_labeled, regime_results)
    """
    if strategy_key not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy_key}'. Choose from: {list(STRATEGIES)}")

    gen_signals, get_p, strat_name = STRATEGIES[strategy_key]

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"\n[regime] Loading {ticker} from {source} (start={start})")
    data = load_data(ticker, start=start, end=end, source=source, ohlcv=False)
    print(f"[regime] {len(data)} rows loaded  ({data.index[0].date()} → {data.index[-1].date()})")

    print(f"[regime] Loading benchmark index: {index_ticker}")
    index_data = load_data(index_ticker, start=start, end=end, source="yfinance", ohlcv=False)

    # ── 2. Strategy backtest ──────────────────────────────────────────────────
    print(f"[regime] Generating signals: {strat_name}")
    df = gen_signals(data, **strategy_kwargs)

    print(f"[regime] Running backtest...")
    df = run_backtest(df)

    metrics = calculate_metrics(df)
    params  = get_p(**strategy_kwargs)
    print_summary(metrics, strategy_name=f"{strat_name}  [{ticker}]  (full period)")

    # ── 3. Train or load HMM ──────────────────────────────────────────────────
    if model_path and os.path.exists(model_path):
        print(f"\n[regime] Loading pre-trained model from {model_path}")
        detector = RegimeDetector.load(model_path)
    else:
        print(f"\n[regime] Training HMM regime detector...")
        detector = RegimeDetector(
            n_regimes = 4,
            n_iter    = 1000,
            window    = feature_window,
        )
        detector.fit(data["Close"], index_data["Close"])

        if save_model:
            detector.save(save_model)

    # ── 4. Regime analysis ────────────────────────────────────────────────────
    print(f"\n[regime] Analyzing strategy performance across regimes...")
    analyzer = RegimeAnalyzer(detector, index_data["Close"])
    results  = analyzer.analyze(df, data["Close"])

    # ── 5. Print regime report ────────────────────────────────────────────────
    analyzer.print_regime_report(results, strategy_name=f"{strat_name}  [{ticker}]")

    # ── 6. Save to SQLite ─────────────────────────────────────────────────────
    if save:
        trades_df = _extract_trades(df)
        run_id = save_run(
            strategy   = f"{strat_name}_regime",
            ticker     = ticker,
            metrics    = metrics,
            params     = {**params, "index_ticker": index_ticker, "feature_window": feature_window},
            start_date = str(data.index[0].date()),
            end_date   = str(data.index[-1].date()),
            trades_df  = trades_df,
            notes      = notes or "Regime analysis run — HMM 4-state",
        )
        print(f"[regime] Full-period run saved → ID {run_id}")

    return df, results


def run_all_strategies_regime(
    ticker,
    start,
    end          = None,
    source       = "yfinance",
    index_ticker = DEFAULT_INDEX,
    model_path   = None,
    save_model   = DEFAULT_MODEL,
    **strategy_kwargs,
):
    """
    Train ONE shared regime model, then run all registered strategies
    (sma, rsi, kalman) through it and print a side-by-side comparison.
    """
    print(f"\n{'='*60}")
    print(f"  Regime Analysis — All Strategies  [{ticker}]")
    print(f"{'='*60}")

    data       = load_data(ticker, start=start, end=end, source=source, ohlcv=False)
    index_data = load_data(index_ticker, start=start, end=end, source="yfinance", ohlcv=False)

    # Train or load one shared HMM
    if model_path and os.path.exists(model_path):
        print(f"\n[regime] Loading shared model from {model_path}")
        detector = RegimeDetector.load(model_path)
    else:
        print(f"\n[regime] Training shared HMM regime detector...")
        detector = RegimeDetector(n_regimes=4, n_iter=1000, window=DEFAULT_WINDOW)
        detector.fit(data["Close"], index_data["Close"])
        if save_model:
            os.makedirs(os.path.dirname(save_model) if os.path.dirname(save_model) else ".", exist_ok=True)
            detector.save(save_model)

    analyzer    = RegimeAnalyzer(detector, index_data["Close"])
    results_map = {}

    for strat_key, (gen_signals, get_p, strat_name) in STRATEGIES.items():
        print(f"\n{'─'*40}")
        print(f"  Running: {strat_name}")
        df      = gen_signals(data, **strategy_kwargs.get(strat_key, {}))
        df      = run_backtest(df)
        results = analyzer.analyze(df, data["Close"])
        results_map[strat_name] = results
        analyzer.print_regime_report(results, strategy_name=f"{strat_name} [{ticker}]")

    analyzer.compare_strategies(results_map)

    return results_map


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AlphaByProcess — Regime Analysis Runner")
    parser.add_argument("--strategy",   default="sma",         help="Strategy key: sma | rsi | kalman")
    parser.add_argument("--ticker",     default="SPY",         help="Ticker symbol")
    parser.add_argument("--start",      default="2015-01-01",  help="Start date")
    parser.add_argument("--end",        default=None,          help="End date (optional)")
    parser.add_argument("--source",     default="yfinance",    help="Data source")
    parser.add_argument("--index",      default=DEFAULT_INDEX, help="Benchmark index ticker")
    parser.add_argument("--window",     default=20, type=int,  help="Feature rolling window (days)")
    parser.add_argument("--load-model", default=None,          help="Path to load pre-trained HMM model")
    parser.add_argument("--save-model", default=None,          help="Path to save trained HMM model")
    parser.add_argument("--compare-strategies", action="store_true",
                        help="Run all strategies and compare across regimes")
    parser.add_argument("--no-save",    action="store_true",   help="Don't save results to DB")
    parser.add_argument("--notes",      default=None,          help="Research note")

    # Kalman-specific knobs (passed through as strategy_kwargs)
    parser.add_argument("--obs-noise",  default=1.0,  type=float, help="Kalman R (default 1.0)")
    parser.add_argument("--proc-noise", default=0.01, type=float, help="Kalman Q (default 0.01)")
    parser.add_argument("--entry-z",    default=1.5,  type=float, help="Kalman entry z (default 1.5)")
    parser.add_argument("--exit-z",     default=0.3,  type=float, help="Kalman exit z (default 0.3)")
    parser.add_argument("--stop-z",     default=3.5,  type=float, help="Kalman stop z (default 3.5)")

    args = parser.parse_args()

    # Build strategy_kwargs — only injected when strategy is kalman
    kalman_kwargs = {}
    if args.strategy == "kalman":
        kalman_kwargs = dict(
            obs_noise_var  = args.obs_noise,
            proc_noise_var = args.proc_noise,
            entry_z        = args.entry_z,
            exit_z         = args.exit_z,
            stop_loss_z    = args.stop_z,
        )

    if args.compare_strategies:
        run_all_strategies_regime(
            ticker       = args.ticker,
            start        = args.start,
            end          = args.end,
            source       = args.source,
            index_ticker = args.index,
            model_path   = args.load_model,
            save_model   = args.save_model or DEFAULT_MODEL,
        )
    else:
        run_regime_analysis(
            strategy_key   = args.strategy,
            ticker         = args.ticker,
            start          = args.start,
            end            = args.end,
            source         = args.source,
            index_ticker   = args.index,
            feature_window = args.window,
            save           = not args.no_save,
            notes          = args.notes,
            model_path     = args.load_model,
            save_model     = args.save_model,
            **kalman_kwargs,
        )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("  AlphaByProcess — Regime Detection Demo")
        print("  SPY 2015→2024 | SMA + RSI + Kalman across regimes")
        print("=" * 60)

        run_all_strategies_regime(
            ticker     = "SPY",
            start      = "2015-01-01",
            save_model = DEFAULT_MODEL,
        )

        print("\n── All saved runs (including regime runs) ──")
        compare_strategies()
    else:
        main()

        