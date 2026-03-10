"""
run.py — Main research orchestrator for AlphaByProcess framework.

Usage:
    # Run a single strategy
    python run.py

    # Compare all saved runs
    python run.py --compare

    # Run with custom params
    python run.py --strategy rsi --ticker BTC-USD --start 2020-01-01
"""

import sys
import os
import argparse

# Make sure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader  import load_data
from utils.performance  import calculate_metrics, print_summary, _extract_trades
from utils.store        import save_run, compare_strategies
from backtests.engine   import run_backtest

# ── Strategy registry ──────────────────────────────────────────────────────────
from strategies.sma_crossover      import generate_signals as sma_signals,      get_params as sma_params,      STRATEGY_NAME as SMA_NAME
from strategies.rsi_mean_reversion import generate_signals as rsi_signals,      get_params as rsi_params,      STRATEGY_NAME as RSI_NAME

STRATEGIES = {
    "sma": (sma_signals, sma_params, SMA_NAME),
    "rsi": (rsi_signals, rsi_params, RSI_NAME),
}
# ──────────────────────────────────────────────────────────────────────────────


def run_strategy(strategy_key, ticker, start, end=None,
                 source="yfinance", save=True, notes=None, **strategy_kwargs):
    """
    Full pipeline: load → signal → backtest → metrics → store → print.

    Args:
        strategy_key    : "sma" | "rsi"
        ticker          : e.g. "BTC-USD", "AAPL"
        start           : start date string
        end             : end date string (optional)
        source          : "yfinance" | "binance" | "csv"
        save            : whether to persist results to SQLite
        notes           : optional research note to attach to this run
        **strategy_kwargs: passed through to generate_signals()

    Returns:
        (df_result, metrics_dict)
    """
    if strategy_key not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy_key}'. Choose from: {list(STRATEGIES)}")

    gen_signals, get_p, strat_name = STRATEGIES[strategy_key]

    print(f"\n[run] Loading {ticker} from {source} (start={start})")
    data = load_data(ticker, start=start, end=end, source=source, ohlcv=False)
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


def main():
    parser = argparse.ArgumentParser(description="AlphaByProcess — Backtest Runner")
    parser.add_argument("--strategy", default="sma",        help="Strategy key: sma | rsi")
    parser.add_argument("--ticker",   default="BTC-USD",    help="Ticker symbol")
    parser.add_argument("--start",    default="2020-01-01", help="Start date")
    parser.add_argument("--end",      default=None,         help="End date (optional)")
    parser.add_argument("--source",   default="yfinance",   help="Data source: yfinance | binance | csv")
    parser.add_argument("--compare",  action="store_true",  help="Print comparison of all saved runs")
    parser.add_argument("--no-save",  action="store_true",  help="Don't save results to DB")
    parser.add_argument("--notes",    default=None,         help="Research note to attach")
    args = parser.parse_args()

    if args.compare:
        compare_strategies()
        return

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
    # ── Quick demo: run both strategies on BTC and compare ────────────────────
    if len(sys.argv) == 1:
        print("=" * 55)
        print("  AlphaByProcess — Running demo: BTC-USD 2020→2024")
        print("=" * 55)

        run_strategy("sma", "BTC-USD", start="2020-01-01",
                     notes="Baseline SMA run", short_window=20, long_window=50)

        run_strategy("rsi", "BTC-USD", start="2020-01-01",
                     notes="Baseline RSI mean reversion run")

        print("\n── Comparison of all saved runs ──")
        compare_strategies()
    else:
        main()