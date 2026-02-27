# Import necessary packages
import ccxt        # for accessing Binance API
import pandas as pd  # for working with DataFrames
import time        # for adding delays between API calls

# Define a function to collect OHLCV data from Binance
def crypto_data(symbol='BTC/USDT', timeframe='1d', start_date="2025-04-28T00:00:00Z"):
    # Initialize Binance exchange object
    binance = ccxt.binance()

    # Convert the start date to a timestamp in milliseconds (required by Binance API)
    since = binance.parse8601(start_date)

    # Initialize empty list to store all fetched OHLCV data
    all_data = []

    # Binance allows max 1000 candles per request
    limit = 1000

    # Loop to repeatedly fetch data in 1000-candle chunks until no more data
    while True:
        # Fetch a batch of OHLCV data
        data = binance.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)

        # If no data is returned, we’ve reached the end -> break the loop
        if not data:
            break

        # Append the newly fetched data to our list
        all_data += data

        # Update the starting point to the timestamp after the last returned candle
        since = data[-1][0] + 1

        # Wait 1 second to avoid hitting Binance's rate limits
        time.sleep(1)

        # If Binance returned fewer than 1000 rows, this was the last batch -> break the loop
        if len(data) < limit:
            break

    # Create a DataFrame from the collected data
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

    # Convert the timestamp column from milliseconds to readable datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Set the timestamp as the DataFrame index
    df.set_index("timestamp", inplace=True)

    # Return the resulting DataFrame
    return df

if __name__ == "__main__":
    # Get full BTC/USDT data from August 2017 onward
    df = crypto_data('BTC/USDT', '1d', start_date="2025-04-28T00:00:00Z")
    
    # Save to CSV
    df.to_csv('data/BTC_USDT_binance_update.csv')
    
    # Print confirmation and data range
    print("✅ Binance OHLCV data saved to data/BTC_USDT_binance.csv")
    print("Data range:", df.index.min(), "to", df.index.max())
