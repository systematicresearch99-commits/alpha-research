import pandas as pd

STRATEGY_NAME = "Oil_Shock_Short"


def generate_signals(data, shock_col="daily_shock", hold_days=3):
    """
    Event-driven short strategy based on oil price shocks.

    Signal logic:
        short (position=-1) for `hold_days` days after an oil shock
        flat  (position=0)  otherwise

    Position is shifted by 1 bar to prevent lookahead bias
    (shock detected on close of day N → short executes from day N+1).

    Args:
        data      : DataFrame with 'Close' (S&P 500) and shock_col columns
        shock_col : column name flagging shock events (0 or 1)
        hold_days : number of days to hold the short after a shock

    Returns:
        DataFrame with signal and position columns added
    """
    df = data.copy()

    # Build signal: -1 for hold_days after each shock, 0 otherwise
    df["signal"] = 0

    shock_dates = df.index[df[shock_col] == 1]

    for shock_date in shock_dates:
        # Find integer location of shock date
        loc = df.index.get_loc(shock_date)

        # Mark the next hold_days bars as short signal
        start = loc + 1
        end   = min(loc + 1 + hold_days, len(df))
        df.iloc[start:end, df.columns.get_loc("signal")] = -1

    # Shift 1 to avoid lookahead — signal on day N → position on day N+1
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


def get_params(shock_col="daily_shock", hold_days=3):
    return {"shock_col": shock_col, "hold_days": hold_days}

    