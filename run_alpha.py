"""
run_alpha.py — Alpha Model Research Runner
==========================================
Runs the full signal-based alpha pipeline:
    raw data → features → alpha model → position → backtest

This is fundamentally different from run.py:
    run.py      : fixed rules  → discrete positions  → backtest
    run_alpha.py: continuous signals → OLS model → sized positions → backtest

Key guarantee — NO lookahead bias:
    Data is split into train/test. The model is fit ONLY on training data.
    Performance is evaluated ONLY on the test period the model never saw.

Usage
-----
    # Basic run — default 70/30 train/test split
    python run_alpha.py --ticker BTC-USD --start 2018-01-01

    # Custom split
    python run_alpha.py --ticker BTC-USD --start 2018-01-01 --train-pct 0.6

    # Walk-forward (re-fits model on rolling windows)
    python run_alpha.py --ticker BTC-USD --start 2018-01-01 --walk-forward

    # Multiple tickers
    python run_alpha.py --tickers BTC-USD ETH-USD SOL-USD --start 2018-01-01 --overlay

    # Compare position bridge modes
    python run_alpha.py --ticker BTC-USD --start 2018-01-01 --compare-modes

    # Long only
    python run_alpha.py --ticker BTC-USD --start 2018-01-01 --long-only
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader  import load_data
from utils.performance  import calculate_metrics, print_summary, _extract_trades
from utils.store        import save_run, compare_strategies
from utils.plotting     import plot_run, plot_overlay, block
from backtests.engine   import run_backtest

from features.feature_engine  import extract_features, get_feature_cols
from features.alpha_model     import AlphaModel
from features.position_bridge import apply as bridge_apply, get_params as bridge_params


# ── Core pipeline ──────────────────────────────────────────────────────────────

def run_alpha(
    ticker,
    start,
    end           = None,
    source        = "yfinance",
    train_pct     = 0.7,
    bridge_mode   = "continuous",
    dead_zone     = 0.1,
    long_only     = False,
    forward_periods = 1,
    save          = True,
    notes         = None,
    show_chart    = True,
):
    """
    Full alpha pipeline: load → features → fit → predict → bridge → backtest.

    Args:
        ticker          : ticker symbol
        start           : start date string
        end             : end date string (optional)
        source          : "yfinance" | "binance" | "csv"
        train_pct       : fraction of data used for training (default 0.7)
        bridge_mode     : "continuous" | "discrete" | "tiered"
        dead_zone       : alpha score dead zone threshold (default 0.1)
        long_only       : only take long positions (default False)
        forward_periods : return horizon for OLS target (default 1 bar)
        save            : persist to SQLite (default True)
        notes           : research note
        show_chart      : show chart (default True)

    Returns:
        (model, df_test, metrics)
    """
    print(f"\n{'='*60}")
    print(f"  Alpha Pipeline  —  {ticker}")
    print(f"  Train: {train_pct*100:.0f}%  |  Test: {(1-train_pct)*100:.0f}%")
    print(f"{'='*60}")

    # ── 1. Load ────────────────────────────────────────────────────────────────
    print(f"\n[alpha] Loading {ticker} (start={start})")
    data = load_data(ticker, start=start, end=end, source=source, ohlcv=True)
    print(f"[alpha] {len(data)} rows  ({data.index[0].date()} → {data.index[-1].date()})")

    # ── 2. Feature extraction ──────────────────────────────────────────────────
    print(f"[alpha] Extracting features...")
    df = extract_features(data)

    feature_cols = get_feature_cols(df)
    print(f"[alpha] Features: {', '.join(feature_cols)}")

    # ── 3. Train/test split ────────────────────────────────────────────────────
    split_idx  = int(len(df) * train_pct)
    train_df   = df.iloc[:split_idx]
    test_df    = df.iloc[split_idx:]
    split_date = df.index[split_idx].date()

    print(f"[alpha] Train: {train_df.index[0].date()} → {train_df.index[-1].date()}  "
          f"({len(train_df)} bars)")
    print(f"[alpha] Test:  {test_df.index[0].date()}  → {test_df.index[-1].date()}  "
          f"({len(test_df)} bars)")

    # ── 4. Fit model ───────────────────────────────────────────────────────────
    print(f"\n[alpha] Fitting OLS alpha model on training data...")
    model = AlphaModel(forward_periods=forward_periods)
    model.fit(train_df)
    model.print_weights()

    print(f"  Feature importance:")
    for feat, imp in model.feature_importance().items():
        bar = "█" * int(imp * 30)
        print(f"    {feat:<20} {bar:<30} {imp*100:.1f}%")

    # ── 5. Predict on full data (train + test) ─────────────────────────────────
    print(f"\n[alpha] Generating alpha scores...")
    df_scored = model.predict(df)

    # ── 6. Position bridge ─────────────────────────────────────────────────────
    print(f"[alpha] Applying position bridge (mode={bridge_mode}, dead_zone={dead_zone})")
    df_pos = bridge_apply(
        df_scored,
        mode       = bridge_mode,
        dead_zone  = dead_zone,
        long_only  = long_only,
    )

    # ── 7. Backtest — TEST PERIOD ONLY ────────────────────────────────────────
    df_test_only = df_pos.iloc[split_idx:].copy()
    df_test_only = run_backtest(df_test_only)
    metrics = calculate_metrics(df_test_only)

    print_summary(metrics, strategy_name=f"Alpha Model  [{ticker}]  (OOS test period)")

    # Also compute in-sample for comparison
    df_train_bt = df_pos.iloc[:split_idx].copy()
    df_train_bt = run_backtest(df_train_bt)
    train_m     = calculate_metrics(df_train_bt)

    print(f"\n  In-sample vs Out-of-sample comparison:")
    print(f"  {'Metric':<22}  {'In-Sample':>12}  {'Out-of-Sample':>14}")
    print(f"  {'─'*52}")
    for m in ["Sharpe Ratio", "Total Return", "Max Drawdown", "Win Rate"]:
        is_val  = train_m.get(m, float("nan"))
        oos_val = metrics.get(m, float("nan"))
        is_pct  = m in ["Total Return", "Max Drawdown", "Win Rate"]
        fmt     = lambda v: f"{v*100:.2f}%" if is_pct else f"{v:.4f}"
        decay   = ""
        if not (np.isnan(is_val) or np.isnan(oos_val)):
            if m == "Sharpe Ratio":
                decay = f"  {'↓' if oos_val < is_val else '↑'} {abs(oos_val - is_val):.3f}"
        print(f"  {m:<22}  {fmt(is_val):>12}  {fmt(oos_val):>14}{decay}")

    # ── 8. Chart ───────────────────────────────────────────────────────────────
    if show_chart:
        chart_title = (f"Alpha Model  |  {ticker}  |  "
                       f"Test: {split_date} → {data.index[-1].date()}")
        plot_run(df_test_only, title=chart_title)

    # ── 9. Save ────────────────────────────────────────────────────────────────
    if save:
        trades_df = _extract_trades(df_test_only)
        params    = {
            "train_pct":      train_pct,
            "bridge_mode":    bridge_mode,
            "dead_zone":      dead_zone,
            "long_only":      long_only,
            "forward_periods": forward_periods,
            "train_r2":       round(model.train_r2, 4),
            "train_ic":       round(model.train_ic, 4),
            "features":       feature_cols,
        }
        run_id = save_run(
            strategy   = "AlphaModel_OLS",
            ticker     = ticker,
            metrics    = metrics,
            params     = params,
            start_date = str(test_df.index[0].date()),
            end_date   = str(test_df.index[-1].date()),
            trades_df  = trades_df,
            notes      = notes or f"OLS alpha model — test period only",
        )
        print(f"\n[alpha] Run saved → ID {run_id}")

    return model, df_test_only, metrics


def run_walk_forward_alpha(
    ticker, start, end=None, source="yfinance",
    n_splits=5, test_pct=0.2, bridge_mode="continuous",
    dead_zone=0.1, long_only=False, overlay=False,
):
    """
    Walk-forward alpha pipeline.
    Re-fits the OLS model on each training window.
    Evaluates only on unseen test windows.
    """
    print(f"\n{'='*60}")
    print(f"  Walk-Forward Alpha  —  {ticker}  ({n_splits} splits)")
    print(f"{'='*60}")

    data = load_data(ticker, start=start, end=end, source=source, ohlcv=True)
    df   = extract_features(data)
    n    = len(df)

    # Build windows
    usable_start = int(n * 0.4)
    test_size    = int(n * test_pct)
    step         = max(test_size, (n - usable_start - test_size) // max(n_splits - 1, 1))

    windows = []
    for i in range(n_splits):
        te_s = usable_start + i * step
        te_e = te_s + test_size
        if te_e > n:
            break
        windows.append((0, te_s, te_s, te_e))

    window_metrics = []
    oos_pieces     = []

    for idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        train_df = df.iloc[tr_s:tr_e]
        test_df  = df.iloc[te_s:te_e].copy()

        try:
            model    = AlphaModel()
            model.fit(train_df)
            scored   = model.predict(test_df)
            positioned = bridge_apply(scored, mode=bridge_mode,
                                      dead_zone=dead_zone, long_only=long_only)
            bt       = run_backtest(positioned)
            m        = calculate_metrics(bt)
            window_metrics.append({
                "window":      idx + 1,
                "test_start":  df.index[te_s].date(),
                "test_end":    df.index[te_e - 1].date(),
                "Sharpe":      m.get("Sharpe Ratio", np.nan),
                "Return":      m.get("Total Return", np.nan),
                "MaxDD":       m.get("Max Drawdown", np.nan),
                "train_r2":    model.train_r2,
                "train_ic":    model.train_ic,
            })
            oos_pieces.append(bt)
        except Exception as e:
            print(f"  [wf-alpha] Window {idx+1} failed: {e}")

    # Print per-window results
    print(f"\n  Walk-Forward OOS Results")
    print(f"  {'─'*65}")
    print(f"  {'Win':>3}  {'Test Start':>12}  {'Test End':>12}  "
          f"{'Sharpe':>7}  {'Return':>8}  {'MaxDD':>8}  {'R²':>6}  {'IC':>6}")
    print(f"  {'─'*65}")

    for w in window_metrics:
        flag = "✓" if w["Sharpe"] > 0 else "✗"
        print(f"  {w['window']:>3}  {str(w['test_start']):>12}  {str(w['test_end']):>12}  "
              f"{w['Sharpe']:>7.3f}  {w['Return']*100:>7.1f}%  "
              f"{w['MaxDD']*100:>7.1f}%  {w['train_r2']:>6.3f}  {w['train_ic']:>6.3f}  {flag}")

    sharpes    = [w["Sharpe"] for w in window_metrics if not np.isnan(w["Sharpe"])]
    consist    = sum(1 for s in sharpes if s > 0) / len(sharpes) if sharpes else 0
    avg_sharpe = np.mean(sharpes) if sharpes else np.nan

    print(f"\n  Consistency:  {consist*100:.0f}%  ({sum(1 for s in sharpes if s > 0)}/{len(sharpes)} profitable)")
    print(f"  Avg OOS Sharpe: {avg_sharpe:.3f}")

    # Chart — chain OOS equity curves
    if oos_pieces and not overlay:
        from utils.plotting import _plot_equity
        running = 1.0
        fig_data = []
        for bt in oos_pieces:
            eq     = bt["equity_curve"].dropna()
            scaled = eq * running
            running = scaled.iloc[-1]
            fig_data.append(scaled)
        chained = pd.concat(fig_data)
        fake_df = oos_pieces[-1].copy()
        fake_df = fake_df.loc[chained.index]
        fake_df["equity_curve"] = chained
        plot_run(fake_df, title=f"Walk-Forward Alpha  |  {ticker}  |  OOS chained equity")

    return window_metrics, oos_pieces


def run_compare_modes(ticker, start, end=None, source="yfinance",
                      train_pct=0.7, dead_zone=0.1):
    """
    Compare all three position bridge modes side by side.
    Uses the same fitted model for each — only the bridge changes.
    """
    print(f"\n[alpha] Comparing bridge modes on {ticker}...")

    data = load_data(ticker, start=start, end=end, source=source, ohlcv=True)
    df   = extract_features(data)

    split_idx = int(len(df) * train_pct)
    model     = AlphaModel()
    model.fit(df.iloc[:split_idx])
    model.print_weights()

    df_scored = model.predict(df)
    results   = {}

    for mode in ["continuous", "discrete", "tiered"]:
        df_pos  = bridge_apply(df_scored, mode=mode, dead_zone=dead_zone)
        df_test = df_pos.iloc[split_idx:].copy()
        df_test = run_backtest(df_test)
        results[f"bridge:{mode}"] = df_test
        m = calculate_metrics(df_test)
        print(f"\n  {mode:<15}  Sharpe={m.get('Sharpe Ratio', float('nan')):>6.3f}  "
              f"Return={m.get('Total Return', 0)*100:>7.2f}%  "
              f"MaxDD={m.get('Max Drawdown', 0)*100:>7.2f}%")

    plot_overlay(results,
                 title=f"Alpha Bridge Mode Comparison  |  {ticker}")
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AlphaByProcess — Alpha Model Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_alpha.py --ticker BTC-USD --start 2018-01-01
  python run_alpha.py --ticker BTC-USD --start 2018-01-01 --train-pct 0.6
  python run_alpha.py --ticker BTC-USD --start 2018-01-01 --walk-forward
  python run_alpha.py --tickers BTC-USD ETH-USD --start 2018-01-01 --overlay
  python run_alpha.py --ticker BTC-USD --start 2018-01-01 --compare-modes
        """
    )

    ticker_group = parser.add_mutually_exclusive_group()
    ticker_group.add_argument("--ticker",  default=None,        help="Single ticker")
    ticker_group.add_argument("--tickers", nargs="+",           help="Multiple tickers")

    parser.add_argument("--start",         default="2018-01-01")
    parser.add_argument("--end",           default=None)
    parser.add_argument("--source",        default="yfinance")
    parser.add_argument("--train-pct",     default=0.7,  type=float, help="Train fraction (default 0.7)")
    parser.add_argument("--bridge-mode",   default="continuous",     help="continuous | discrete | tiered")
    parser.add_argument("--dead-zone",     default=0.1,  type=float, help="Alpha score dead zone (default 0.1)")
    parser.add_argument("--long-only",     action="store_true",      help="Long only positions")
    parser.add_argument("--forward",       default=1,    type=int,   help="Forward return periods (default 1)")
    parser.add_argument("--walk-forward",  action="store_true",      help="Walk-forward mode")
    parser.add_argument("--splits",        default=5,    type=int,   help="Walk-forward splits")
    parser.add_argument("--compare-modes", action="store_true",      help="Compare bridge modes")
    parser.add_argument("--overlay",       action="store_true",      help="Overlay multi-ticker charts")
    parser.add_argument("--no-save",       action="store_true")
    parser.add_argument("--compare-all",   action="store_true",      help="Compare all saved runs in DB")
    parser.add_argument("--notes",         default=None)

    args = parser.parse_args()

    if args.compare_all:
        compare_strategies()
        return

    tickers = args.tickers or ([args.ticker] if args.ticker else ["BTC-USD"])

    if args.compare_modes:
        for ticker in tickers:
            run_compare_modes(ticker, args.start, args.end, args.source,
                              args.train_pct, args.dead_zone)
        block()
        return

    if args.walk_forward:
        results_map = {}
        for ticker in tickers:
            wf_metrics, _ = run_walk_forward_alpha(
                ticker, args.start, args.end, args.source,
                n_splits=args.splits, bridge_mode=args.bridge_mode,
                dead_zone=args.dead_zone, long_only=args.long_only,
                overlay=args.overlay,
            )
            results_map[ticker] = wf_metrics
        block()
        return

    # Standard run
    results = {}
    for ticker in tickers:
        model, df_test, metrics = run_alpha(
            ticker          = ticker,
            start           = args.start,
            end             = args.end,
            source          = args.source,
            train_pct       = args.train_pct,
            bridge_mode     = args.bridge_mode,
            dead_zone       = args.dead_zone,
            long_only       = args.long_only,
            forward_periods = args.forward,
            save            = not args.no_save,
            notes           = args.notes,
            show_chart      = not args.overlay,
        )
        results[ticker] = df_test

    if args.overlay and len(results) > 1:
        plot_overlay(results, title=f"Alpha Model  |  Multi-Ticker Comparison")

    block()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("  AlphaByProcess — Alpha Model Demo")
        print("  BTC-USD 2018→2024")
        print("=" * 60)

        # Standard run
        run_alpha("BTC-USD", "2018-01-01", notes="Alpha model baseline")

        # Walk-forward validation
        run_walk_forward_alpha("BTC-USD", "2018-01-01", n_splits=5)

        # Compare bridge modes
        run_compare_modes("BTC-USD", "2018-01-01")

        block()
    else:
        main()

        