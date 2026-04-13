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

    # Compare multiple strategies across regimes
    python run_regime.py --compare-strategies

    # Use a previously saved model (skip retraining)
    python run_regime.py --load-model results/regime_model.pkl

    # Save the trained HMM model
    python run_regime.py --save-model results/regime_model.pkl

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
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader  import load_data
from utils.performance  import calculate_metrics, print_summary, _extract_trades
from utils.store        import save_run, compare_strategies
from utils.plotting     import plot_regime_run, plot_overlay, block
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

from research.regime_detection.hmm_model       import RegimeDetector
from research.regime_detection.regime_analyzer import RegimeAnalyzer

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

DEFAULT_INDEX  = "SPY"
DEFAULT_MODEL  = "results/regime_model.pkl"
DEFAULT_WINDOW = 20


# ── Core single-run pipeline ──────────────────────────────────────────────────

def run_regime_analysis(
    strategy_key, ticker, start, end=None, source="yfinance",
    index_ticker=DEFAULT_INDEX, feature_window=DEFAULT_WINDOW,
    save=True, notes=None, model_path=None, save_model=None,
    show_chart=True, detector=None,
    **strategy_kwargs,
):
    """
    Single strategy × single ticker regime pipeline.
    Pass detector= to reuse an already-trained HMM (avoids retraining).
    Returns (df, results, detector).
    """
    if strategy_key not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy_key}'. Choose from: {list(STRATEGIES)}")

    gen_signals, get_p, strat_name = STRATEGIES[strategy_key]

    print(f"\n[regime] Loading {ticker} from {source} (start={start})")
    data = load_data(ticker, start=start, end=end, source=source, ohlcv=True)
    print(f"[regime] {len(data)} rows loaded  ({data.index[0].date()} → {data.index[-1].date()})")

    print(f"[regime] Loading benchmark index: {index_ticker}")
    index_data = load_data(index_ticker, start=start, end=end, source="yfinance", ohlcv=True)

    print(f"[regime] Generating signals: {strat_name}")
    df = gen_signals(data, **strategy_kwargs)

    print(f"[regime] Running backtest...")
    df = run_backtest(df)

    metrics = calculate_metrics(df)
    params  = get_p(**strategy_kwargs)
    print_summary(metrics, strategy_name=f"{strat_name}  [{ticker}]  (full period)")

    # Train or reuse HMM
    if detector is None:
        if model_path and os.path.exists(model_path):
            print(f"\n[regime] Loading pre-trained model from {model_path}")
            detector = RegimeDetector.load(model_path)
        else:
            print(f"\n[regime] Training HMM regime detector...")
            detector = RegimeDetector(n_regimes=4, n_iter=1000, window=feature_window)
            detector.fit(data["Close"], index_data["Close"])
            if save_model:
                detector.save(save_model)

    analyzer = RegimeAnalyzer(detector, index_data["Close"])
    results  = analyzer.analyze(df, data["Close"])
    analyzer.print_regime_report(results, strategy_name=f"{strat_name}  [{ticker}]")

    if hasattr(analyzer, "attach_regimes"):
        df = analyzer.attach_regimes(df, data["Close"])

    if show_chart:
        chart_title = f"{strat_name}  |  {ticker}  |  Regimes  |  {data.index[0].date()} → {data.index[-1].date()}"
        plot_regime_run(df, results, title=chart_title)

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
        print(f"[regime] Run saved → ID {run_id}")

    return df, results, detector


# ── Multi-run pipelines ───────────────────────────────────────────────────────

def run_multi_strategy_regime(
    strategy_keys, ticker, start, end=None, source="yfinance",
    index_ticker=DEFAULT_INDEX, feature_window=DEFAULT_WINDOW,
    save=True, notes=None, model_path=None, save_model=DEFAULT_MODEL,
    overlay=False, **strategy_kwargs,
):
    """Multiple strategies × one ticker — shares one trained HMM."""
    print(f"\n[regime] Multi-strategy regime — {len(strategy_keys)} strategies on {ticker}")

    # Load data + train one shared HMM
    data       = load_data(ticker,       start=start, end=end, source=source,     ohlcv=True)
    index_data = load_data(index_ticker, start=start, end=end, source="yfinance", ohlcv=True)

    if model_path and os.path.exists(model_path):
        detector = RegimeDetector.load(model_path)
    else:
        print(f"[regime] Training shared HMM...")
        detector = RegimeDetector(n_regimes=4, n_iter=1000, window=feature_window)
        detector.fit(data["Close"], index_data["Close"])
        if save_model:
            os.makedirs(os.path.dirname(save_model) if os.path.dirname(save_model) else ".", exist_ok=True)
            detector.save(save_model)

    results_map = {}
    for key in strategy_keys:
        df, results, _ = run_regime_analysis(
            key, ticker, start, end=end, source=source,
            index_ticker=index_ticker, feature_window=feature_window,
            save=save, notes=notes, detector=detector,
            show_chart=not overlay,
            **strategy_kwargs.get(key, {}),
        )
        results_map[STRATEGIES[key][2]] = df

    if overlay and results_map:
        plot_overlay(results_map,
                     title=f"Strategy Comparison  |  {ticker}  |  Regimes  |  {start} → {end or 'today'}")

    return results_map


def run_multi_ticker_regime(
    strategy_key, tickers, start, end=None, source="yfinance",
    index_ticker=DEFAULT_INDEX, feature_window=DEFAULT_WINDOW,
    save=True, notes=None, overlay=False, **strategy_kwargs,
):
    """One strategy × multiple tickers — trains separate HMM per ticker."""
    strat_name = STRATEGIES[strategy_key][2]
    print(f"\n[regime] Multi-ticker regime — {strat_name} on {len(tickers)} tickers")

    results_map = {}
    for ticker in tickers:
        df, results, _ = run_regime_analysis(
            strategy_key, ticker, start, end=end, source=source,
            index_ticker=index_ticker, feature_window=feature_window,
            save=save, notes=notes,
            show_chart=not overlay,
            **strategy_kwargs,
        )
        results_map[ticker] = df

    if overlay and results_map:
        plot_overlay(results_map,
                     title=f"{strat_name}  |  Ticker Comparison  |  Regimes  |  {start} → {end or 'today'}")

    return results_map


def run_matrix_regime(
    strategy_keys, tickers, start, end=None, source="yfinance",
    index_ticker=DEFAULT_INDEX, feature_window=DEFAULT_WINDOW,
    save=True, notes=None, overlay=False, **strategy_kwargs,
):
    """Multiple strategies × multiple tickers — overlay per ticker if enabled."""
    print(f"\n[regime] Matrix regime — {len(strategy_keys)} strategies × {len(tickers)} tickers")

    all_results = {}
    for ticker in tickers:
        ticker_map = {}
        data       = load_data(ticker,       start=start, end=end, source=source,     ohlcv=True)
        index_data = load_data(index_ticker, start=start, end=end, source="yfinance", ohlcv=True)

        print(f"[regime] Training shared HMM for {ticker}...")
        detector = RegimeDetector(n_regimes=4, n_iter=1000, window=feature_window)
        detector.fit(data["Close"], index_data["Close"])

        for key in strategy_keys:
            df, results, _ = run_regime_analysis(
                key, ticker, start, end=end, source=source,
                index_ticker=index_ticker, feature_window=feature_window,
                save=save, notes=notes, detector=detector,
                show_chart=not overlay,
                **strategy_kwargs.get(key, {}),
            )
            label = STRATEGIES[key][2]
            ticker_map[label] = df
            all_results[f"{label} | {ticker}"] = df

        if overlay and ticker_map:
            plot_overlay(ticker_map,
                         title=f"Strategy Comparison  |  {ticker}  |  Regimes")

    return all_results


def run_all_strategies_regime(
    ticker, start, end=None, source="yfinance",
    index_ticker=DEFAULT_INDEX, model_path=None, save_model=DEFAULT_MODEL,
    strategy_subset=None, **strategy_kwargs,
):
    """Original compare-all runner — trains one shared HMM, runs all strategies."""
    subset  = strategy_subset or list(STRATEGIES.keys())
    invalid = [k for k in subset if k not in STRATEGIES]
    if invalid:
        raise ValueError(f"Unknown strategy keys: {invalid}")

    print(f"\n{'='*65}")
    print(f"  Regime Analysis — {len(subset)} Strategies  [{ticker}]")
    print(f"{'='*65}")

    data       = load_data(ticker,       start=start, end=end, source=source,     ohlcv=True)
    index_data = load_data(index_ticker, start=start, end=end, source="yfinance", ohlcv=True)

    if model_path and os.path.exists(model_path):
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

    for strat_key in subset:
        gen_signals, get_p, strat_name = STRATEGIES[strat_key]
        per_strat_kwargs = strategy_kwargs.get(strat_key, {})
        print(f"\n{'─'*45}\n  Strategy: {strat_name}")

        df      = gen_signals(data, **per_strat_kwargs)
        df      = run_backtest(df)
        results = analyzer.analyze(df, data["Close"])
        results_map[strat_name] = results
        analyzer.print_regime_report(results, strategy_name=f"{strat_name} [{ticker}]")

        if hasattr(analyzer, "attach_regimes"):
            df = analyzer.attach_regimes(df, data["Close"])
        plot_regime_run(df, results, title=f"{strat_name}  |  {ticker}  |  Regimes")

    print(f"\n{'='*65}\n  Cross-Strategy Comparison  [{ticker}]\n{'='*65}")
    analyzer.compare_strategies(results_map)
    return results_map


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AlphaByProcess — Regime Analysis Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_regime.py --strategy macd --ticker SPY
  python run_regime.py --strategies sma ema macd --ticker HDFCBANK.NS --index ^NSEI
  python run_regime.py --strategies sma ema macd --ticker HDFCBANK.NS --overlay
  python run_regime.py --strategy macd --tickers HDFCBANK.NS RELIANCE.NS
  python run_regime.py --compare-strategies --subset sma,ema,macd
        """
    )

    strat_group = parser.add_mutually_exclusive_group()
    strat_group.add_argument("--strategy",           default=None,          help="Single strategy key")
    strat_group.add_argument("--strategies",          nargs="+",             help="Multiple strategy keys")
    strat_group.add_argument("--compare-strategies",  action="store_true",   help="Run all strategies")

    ticker_group = parser.add_mutually_exclusive_group()
    ticker_group.add_argument("--ticker",  default=None,   help="Single ticker")
    ticker_group.add_argument("--tickers", nargs="+",      help="Multiple tickers")

    parser.add_argument("--start",      default="2015-01-01")
    parser.add_argument("--end",        default=None)
    parser.add_argument("--source",     default="yfinance")
    parser.add_argument("--index",      default=DEFAULT_INDEX,  help="Benchmark index for HMM")
    parser.add_argument("--window",     default=20,   type=int, help="HMM feature window")
    parser.add_argument("--load-model", default=None)
    parser.add_argument("--save-model", default=None)
    parser.add_argument("--subset",     default=None,            help="Comma-separated keys for --compare-strategies")
    parser.add_argument("--overlay",    action="store_true",     help="Overlay equity curves on one chart")
    parser.add_argument("--no-save",    action="store_true")
    parser.add_argument("--notes",      default=None)
    parser.add_argument("--obs-noise",  default=1.0,  type=float)
    parser.add_argument("--proc-noise", default=0.01, type=float)
    parser.add_argument("--entry-z",    default=1.5,  type=float)
    parser.add_argument("--exit-z",     default=0.3,  type=float)
    parser.add_argument("--stop-z",     default=3.5,  type=float)

    args = parser.parse_args()

    strategies = args.strategies or ([args.strategy] if args.strategy else None)
    tickers    = args.tickers    or ([args.ticker]    if args.ticker    else ["SPY"])
    subset     = [s.strip() for s in args.subset.split(",")] if args.subset else None
    save       = not args.no_save

    kalman_kwargs = {}
    if strategies and "kalman" in strategies:
        kalman_kwargs = dict(
            obs_noise_var=args.obs_noise, proc_noise_var=args.proc_noise,
            entry_z=args.entry_z, exit_z=args.exit_z, stop_loss_z=args.stop_z,
        )

    if args.compare_strategies:
        run_all_strategies_regime(
            ticker=tickers[0], start=args.start, end=args.end,
            source=args.source, index_ticker=args.index,
            model_path=args.load_model,
            save_model=args.save_model or DEFAULT_MODEL,
            strategy_subset=subset,
        )

    elif strategies and len(strategies) > 1 and len(tickers) == 1:
        run_multi_strategy_regime(
            strategies, tickers[0], args.start, args.end,
            args.source, args.index, args.window, save, args.notes,
            args.load_model, args.save_model or DEFAULT_MODEL,
            overlay=args.overlay,
        )

    elif strategies and len(strategies) == 1 and len(tickers) > 1:
        run_multi_ticker_regime(
            strategies[0], tickers, args.start, args.end,
            args.source, args.index, args.window, save, args.notes,
            overlay=args.overlay, **kalman_kwargs,
        )

    elif strategies and len(strategies) > 1 and len(tickers) > 1:
        run_matrix_regime(
            strategies, tickers, args.start, args.end,
            args.source, args.index, args.window, save, args.notes,
            overlay=args.overlay,
        )

    else:
        strat = (strategies or ["sma"])[0]
        run_regime_analysis(
            strat, tickers[0], args.start, args.end,
            args.source, args.index, args.window, save, args.notes,
            args.load_model, args.save_model,
            **kalman_kwargs,
        )

    block()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 65)
        print("  AlphaByProcess — Regime Demo: SPY 2015→2024")
        print("=" * 65)
        run_all_strategies_regime(ticker="SPY", start="2015-01-01", save_model=DEFAULT_MODEL)
        compare_strategies()
        block()
    else:
        main()