"""
store.py — SQLite-backed research log.

Every backtest run is persisted with:
  - strategy name + parameters
  - ticker + date range
  - all performance metrics
  - timestamp of the run

Tables:
  runs    — one row per backtest run
  trades  — individual trade log (optional, linked to run)
"""

import sqlite3
import json
import os
from datetime import datetime

# Default DB path: results/research.db (relative to project root)
_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "research.db"
)


def _get_conn(db_path=None):
    path = db_path or _DEFAULT_DB
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    return conn


def init_db(db_path=None):
    """Create tables if they don't exist."""
    conn = _get_conn(db_path)
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            strategy        TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            start_date      TEXT,
            end_date        TEXT,
            params          TEXT,        -- JSON string of strategy parameters
            total_return    REAL,
            annualized_ret  REAL,
            sharpe          REAL,
            sortino         REAL,
            calmar          REAL,
            max_drawdown    REAL,
            annualized_vol  REAL,
            num_trades      INTEGER,
            win_rate        REAL,
            profit_factor   REAL,
            avg_hold_days   REAL,
            notes           TEXT
        );

        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      INTEGER REFERENCES runs(id),
            entry_date  TEXT,
            exit_date   TEXT,
            pnl         REAL,
            hold_days   REAL
        );
    """)
    conn.commit()
    conn.close()


def save_run(strategy, ticker, metrics, params=None,
             start_date=None, end_date=None,
             trades_df=None, notes=None, db_path=None):
    """
    Persist a backtest run.

    Args:
        strategy  : strategy name string e.g. "SMA_Crossover"
        ticker    : e.g. "BTC-USD"
        metrics   : dict from performance.calculate_metrics()
        params    : dict of strategy parameters e.g. {"short": 20, "long": 50}
        trades_df : optional DataFrame of trades from performance._extract_trades()
        notes     : optional free-text note
        db_path   : optional override for DB location

    Returns:
        run_id (int)
    """
    init_db(db_path)
    conn = _get_conn(db_path)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO runs (
            timestamp, strategy, ticker, start_date, end_date, params,
            total_return, annualized_ret, sharpe, sortino, calmar,
            max_drawdown, annualized_vol, num_trades, win_rate,
            profit_factor, avg_hold_days, notes
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(),
        strategy,
        ticker,
        start_date,
        end_date,
        json.dumps(params) if params else None,
        metrics.get("Total Return"),
        metrics.get("Annualized Return"),
        metrics.get("Sharpe Ratio"),
        metrics.get("Sortino Ratio"),
        metrics.get("Calmar Ratio"),
        metrics.get("Max Drawdown"),
        metrics.get("Annualized Vol"),
        metrics.get("Num Trades"),
        metrics.get("Win Rate"),
        metrics.get("Profit Factor"),
        metrics.get("Avg Hold Days"),
        notes,
    ))

    run_id = cur.lastrowid

    # Save individual trades if provided
    if trades_df is not None and not trades_df.empty:
        for _, t in trades_df.iterrows():
            cur.execute("""
                INSERT INTO trades (run_id, entry_date, exit_date, pnl, hold_days)
                VALUES (?,?,?,?,?)
            """, (
                run_id,
                str(t.get("entry", "")),
                str(t.get("exit", "")),
                t.get("pnl"),
                t.get("hold_days"),
            ))

    conn.commit()
    conn.close()
    print(f"[store] Run saved → ID {run_id}  ({strategy} on {ticker})")
    return run_id


def load_runs(db_path=None):
    """Load all runs as a DataFrame for comparison."""
    import pandas as pd
    init_db(db_path)
    conn = _get_conn(db_path)
    df = pd.read_sql("SELECT * FROM runs ORDER BY timestamp DESC", conn)
    conn.close()
    return df


def compare_strategies(db_path=None):
    """
    Print a ranked comparison table of all saved runs,
    sorted by Sharpe ratio descending.
    """
    df = load_runs(db_path)
    if df.empty:
        print("[store] No runs saved yet.")
        return df

    cols = ["id","timestamp","strategy","ticker","sharpe","total_return",
            "max_drawdown","win_rate","num_trades","annualized_ret"]
    df_show = df[cols].copy()
    df_show = df_show.sort_values("sharpe", ascending=False)

    # Format for display
    pct_cols = ["total_return","max_drawdown","win_rate","annualized_ret"]
    for c in pct_cols:
        df_show[c] = df_show[c].map(lambda x: f"{x*100:.1f}%" if x == x else "—")
    df_show["sharpe"] = df_show["sharpe"].map(lambda x: f"{x:.3f}" if x == x else "—")

    print("\n" + "═"*90)
    print("  STRATEGY COMPARISON  (sorted by Sharpe)")
    print("═"*90)
    print(df_show.to_string(index=False))
    print("═"*90 + "\n")
    return df


def delete_run(run_id, db_path=None):
    """Remove a run and its trades by ID."""
    conn = _get_conn(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM trades WHERE run_id=?", (run_id,))
    cur.execute("DELETE FROM runs WHERE id=?", (run_id,))
    conn.commit()
    conn.close()
    print(f"[store] Run {run_id} deleted.")
    