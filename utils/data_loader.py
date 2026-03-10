import yfinance as yf
import pandas as pd
import os

def load_data(ticker, start="2018-01-01", end=None, source="yfinance", ohlcv=False):
    """
    Load price data for a ticker.

    Args:
        ticker:  e.g. "BTC-USD", "AAPL"
        start:   start date string
        end:     end date string (defaults to today)
        source:  "yfinance" | "binance" | "csv"
        ohlcv:   if True, return full OHLCV; if False, return Close only

    Returns:
        pd.DataFrame
    """
    if source == "yfinance":
        return _load_yfinance(ticker, start, end, ohlcv)
    elif source == "binance":
        return _load_binance(ticker, start, end, ohlcv)
    elif source == "csv":
        return _load_csv(ticker, ohlcv)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'yfinance', 'binance', or 'csv'.")


def _load_yfinance(ticker, start, end, ohlcv):
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No data returned for {ticker} from yfinance.")
    if ohlcv:
        data = data[["Open", "High", "Low", "Close", "Volume"]]
    else:
        data = data[["Close"]]
    # Flatten MultiIndex columns if present (yfinance quirk)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.dropna(inplace=True)
    return data


def _load_binance(ticker, start, end, ohlcv):
    """
    Fetch OHLCV from Binance public API (no API key needed).
    ticker format: "BTCUSDT", "ETHUSDT"
    """
    try:
        import requests
        from datetime import datetime

        base_url = "https://api.binance.com/api/v3/klines"
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms   = int(pd.Timestamp(end).timestamp() * 1000) if end else int(pd.Timestamp.now().timestamp() * 1000)

        all_candles = []
        limit = 1000

        while start_ms < end_ms:
            params = {
                "symbol": ticker.upper(),
                "interval": "1d",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            }
            resp = requests.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
            candles = resp.json()
            if not candles:
                break
            all_candles.extend(candles)
            start_ms = candles[-1][0] + 1

        cols = ["Open time","Open","High","Low","Close","Volume",
                "Close time","Quote vol","Trades","Taker buy base",
                "Taker buy quote","Ignore"]
        df = pd.DataFrame(all_candles, columns=cols)
        df["Date"] = pd.to_datetime(df["Open time"], unit="ms")
        df.set_index("Date", inplace=True)
        df = df[["Open","High","Low","Close","Volume"]].astype(float)
        df.dropna(inplace=True)

        if not ohlcv:
            df = df[["Close"]]
        return df

    except Exception as e:
        print(f"[data_loader] Binance failed ({e}), falling back to yfinance.")
        # Convert Binance ticker format to yfinance (BTCUSDT → BTC-USD)
        yf_ticker = ticker.replace("USDT", "-USD").replace("BTC", "BTC")
        return _load_yfinance(yf_ticker, start, end, ohlcv)


def _load_csv(ticker, ohlcv):
    """
    Load from data/raw/<ticker>.csv
    Expects columns: Date, Open, High, Low, Close, Volume
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "data", "raw", f"{ticker}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}")
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df.dropna(inplace=True)
    if not ohlcv:
        df = df[["Close"]]
    return df