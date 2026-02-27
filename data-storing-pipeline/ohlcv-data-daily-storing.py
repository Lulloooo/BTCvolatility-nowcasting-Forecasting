################
# BACK UP OHLCV DATA (triggered daily by daily_BTCbackup.yml actions)
# This saves OHLCV volatility data from CoinGecko (BTCVolatility-data/volatility_data.csv)
# The saved df is used as backup in the app.py if issues arises in fetching Coingeko API
################

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

#define symbol, currency, path and n-day to fetch
SYMBOL = "bitcoin"
VS_CURRENCY = "usd"
base_dir = Path(__file__).resolve().parents[1]
OUTPUT_PATH = base_dir / "data" / "ohlcv-data-daily-storing" / "ohclv-daily-storing.csv"
 # Number of days to fetch
N_DAYS = 90 

#define a function to fetch 90-days data from CoinGeko
def fetch_ohlcv_coingecko(symbol, vs_currency, n_days=90):
    end = datetime.now()
    start = end - timedelta(days=n_days)

    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart/range"
    params = {
        "vs_currency": vs_currency,
        "from": int(start.timestamp()),
        "to": int(end.timestamp())
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    prices = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
    volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
    df = prices.merge(volumes, on="timestamp")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # OHLC resample
    df = df.resample("1D").agg(
        open=("close", "first"),
        high=("close", "max"),
        low=("close", "min"),
        close=("close", "last"),
        volume=("volume", "sum")
    ).dropna()

    return df

#define a function fetching and storing data
def update_backup():
    '''
    Search for volatility_data, if it exists update it with the last available data.
    If not, create it and store 90-days data.
    '''
    OUTPUT_PATH.parent.mkdir(exist_ok=True, parents=True)

    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH, parse_dates=["timestamp"], index_col="timestamp")
        last_ts = existing.index[-1]
        df_new = fetch_ohlcv_coingecko(SYMBOL, VS_CURRENCY, N_DAYS)
        df = pd.concat([existing, df_new]).drop_duplicates()
    else:
        df = fetch_ohlcv_coingecko(SYMBOL, VS_CURRENCY, N_DAYS)

    df = df.sort_index()
    df.to_csv(OUTPUT_PATH)
    print(f"✅ Saved backup: {len(df)} rows ({df.index.min()} → {df.index.max()})")

#fetch and store last data
if __name__ == "__main__":
    update_backup()
