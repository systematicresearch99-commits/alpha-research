import numpy as np
import pandas as pd


def calculate_metrics(df, periods_per_year=252):
    """
    Compute a full set of performance metrics from a backtest DataFrame.

    Required columns in df:
        - strategy_returns : daily strategy returns (after costs)
        - equity_curve     : cumulative equity (1 + r).cumprod()
        - position         : signal position (1 = long, 0 = flat, -1 = short)

    Returns:
        dict of metrics
    """
    r = df["strategy_returns"].dropna()
    eq = df["equity_curve"].dropna()

    # ── Core returns ──────────────────────────────────────────────
    total_return = eq.iloc[-1] - 1

    n_years = len(r) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    # ── Risk ──────────────────────────────────────────────────────
    annualized_vol = r.std() * np.sqrt(periods_per_year)

    drawdown_series = eq / eq.cummax() - 1
    max_drawdown = drawdown_series.min()

    # ── Risk-adjusted ─────────────────────────────────────────────
    sharpe = (r.mean() * periods_per_year) / (r.std() * np.sqrt(periods_per_year)) \
             if r.std() > 0 else np.nan

    calmar = annualized_return / abs(max_drawdown) \
             if max_drawdown != 0 else np.nan

    # Sortino — downside deviation only
    downside = r[r < 0]
    sortino = (r.mean() * periods_per_year) / (downside.std() * np.sqrt(periods_per_year)) \
              if len(downside) > 0 and downside.std() > 0 else np.nan

    # ── Trade stats ───────────────────────────────────────────────
    trades      = _extract_trades(df)
    n_trades    = len(trades)
    win_rate    = (trades["pnl"] > 0).mean() if n_trades > 0 else np.nan
    avg_win     = trades.loc[trades["pnl"] > 0, "pnl"].mean() if n_trades > 0 else np.nan
    avg_loss    = trades.loc[trades["pnl"] < 0, "pnl"].mean() if n_trades > 0 else np.nan
    profit_factor = abs(avg_win / avg_loss) if (avg_loss and avg_loss != 0) else np.nan
    avg_hold_days = trades["hold_days"].mean() if n_trades > 0 else np.nan

    return {
        # Returns
        "Total Return":        round(total_return, 4),
        "Annualized Return":   round(annualized_return, 4),
        "Annualized Vol":      round(annualized_vol, 4),
        # Risk-adjusted
        "Sharpe Ratio":        round(sharpe, 4),
        "Sortino Ratio":       round(sortino, 4),
        "Calmar Ratio":        round(calmar, 4),
        # Drawdown
        "Max Drawdown":        round(max_drawdown, 4),
        # Trades
        "Num Trades":          n_trades,
        "Win Rate":            round(win_rate, 4) if not np.isnan(win_rate) else np.nan,
        "Avg Win":             round(avg_win, 4)  if avg_win and not np.isnan(avg_win) else np.nan,
        "Avg Loss":            round(avg_loss, 4) if avg_loss and not np.isnan(avg_loss) else np.nan,
        "Profit Factor":       round(profit_factor, 4) if profit_factor and not np.isnan(profit_factor) else np.nan,
        "Avg Hold Days":       round(avg_hold_days, 1) if avg_hold_days and not np.isnan(avg_hold_days) else np.nan,
    }


def _extract_trades(df):
    """
    Extract individual trades from position column.
    A trade = entry when position changes from 0→1 (or flip), exit when it goes back.
    """
    df = df.copy()
    df["pos"] = df["position"].fillna(0)
    df["trade_start"] = df["pos"].diff().fillna(0) != 0

    trades = []
    in_trade = False
    entry_date = None
    entry_price = None

    for date, row in df.iterrows():
        if row["trade_start"] and not in_trade and row["pos"] != 0:
            in_trade = True
            entry_date = date
            entry_price = row["Close"] if "Close" in df.columns else None

        elif row["trade_start"] and in_trade:
            exit_price = row["Close"] if "Close" in df.columns else None
            pnl = (exit_price - entry_price) / entry_price if (entry_price and exit_price) else np.nan
            hold_days = (date - entry_date).days if entry_date else np.nan
            trades.append({"entry": entry_date, "exit": date,
                           "pnl": pnl, "hold_days": hold_days})
            if row["pos"] != 0:
                entry_date = date
                entry_price = exit_price
            else:
                in_trade = False

    return pd.DataFrame(trades) if trades else pd.DataFrame(columns=["entry","exit","pnl","hold_days"])


def print_summary(metrics, strategy_name="Strategy"):
    """Pretty-print metrics to console."""
    line = "─" * 40
    print(f"\n{line}")
    print(f"  {strategy_name}")
    print(line)
    groups = [
        ("Returns",       ["Total Return","Annualized Return","Annualized Vol"]),
        ("Risk-Adjusted", ["Sharpe Ratio","Sortino Ratio","Calmar Ratio"]),
        ("Drawdown",      ["Max Drawdown"]),
        ("Trades",        ["Num Trades","Win Rate","Avg Win","Avg Loss","Profit Factor","Avg Hold Days"]),
    ]
    for group_name, keys in groups:
        print(f"\n  {group_name}")
        for k in keys:
            v = metrics.get(k, "—")
            if isinstance(v, float):
                if "Return" in k or "Vol" in k or "Drawdown" in k or "Rate" in k or "Win" in k or "Loss" in k:
                    print(f"    {k:<22} {v*100:>8.2f}%")
                else:
                    print(f"    {k:<22} {v:>8.4f}")
            else:
                print(f"    {k:<22} {str(v):>8}")
    print(f"\n{line}\n")