#script to merge the dataset fetched

#import the needed packages
import pandas as pd
import random
import fastparquet

#load the dataset
btc = pd.read_csv("data/BTC_USDT_binance.csv") #btc
fear = pd.read_csv("data/fear_greed_index.csv") #fear e greed index
trend = pd.read_csv("data/google_trends.csv") #google trends

#define the starting date
start_date = "2018-02-01"
end_date = "2025-04-28"

#filter the datasets so they are all starting from the same date (February 1st 2018)
btc_filtered = (
    btc[(btc["timestamp"] >= start_date) & (btc["timestamp"] <= end_date)]
    .reset_index(drop = True)
    )

trend_filtered = (
    trend[(trend["timestamp"] >= start_date) & (trend["timestamp"] <= end_date)]
    .reset_index(drop = True)
    )

fear_filtered = (
    fear[(fear["timestamp"] >= start_date) & (fear["timestamp"] <= end_date)]
    .reset_index(drop = True)
    )
#fear_filtered has 4 missing dates. We will forward fill them (3 are consecutives dates)
#spot missing dates
# Generate the full expected date range
full_range = pd.date_range(start=start_date, end=end_date)
# Get the actual dates in your fear dataset
actual_dates = pd.to_datetime(fear_filtered["timestamp"].unique())
# Find which dates are missing
missing_dates = full_range.difference(actual_dates)
# Print them
print("Missing dates in fear_filtered:")
print(missing_dates)
#add the missing date to fear df
# Create a DataFrame with missing dates (empty values)
missing_df = pd.DataFrame({
    "timestamp": missing_dates,
    "fear&greed_index": [None] * len(missing_dates)
})
# Append to the original dataframe
fear_filled = pd.concat([fear_filtered, missing_df], ignore_index=True)
# Ensure all timestamp values are datetime
fear_filled["timestamp"] = pd.to_datetime(fear_filled["timestamp"])
# Now sort safely
fear_filled = fear_filled.sort_values("timestamp").reset_index(drop=True)
# Forward-fill
fear_filled["fear&greed_index"] = fear_filled["fear&greed_index"].ffill()
#check the dfs
print(btc_filtered)
print(trend_filtered)
print(fear_filled)
#save fear_filled
fear_filled.to_csv("data/fearGreed_final.csv", index = False)
print(fear_filled.dtypes)
print(btc_filtered.dtypes)
#merge the datasets together to have a final df
# inner join
merged_1 = pd.merge(btc_filtered, trend_filtered, on='timestamp', how='inner')
#transform into timestamp
merged_1["timestamp"] = pd.to_datetime(merged_1["timestamp"])
#merge with the second df
final_df = pd.merge(merged_1, fear_filled, on = "timestamp", how = "inner")
print(final_df)
#check the mergings
print("BTC rows:", len(btc_filtered))
print("Trend rows:", len(trend_filtered))
print("Fear rows (filled):", len(fear_filled))
print("After first merge:", len(merged_1))
print("Final merged rows:", len(final_df))
#check there are no NaN
print(final_df.isna().sum())
#save the final df
final_df.to_csv("data/crypto_volatility_raw.csv", index=False)
final_df.to_parquet("data/crypto_volatility_raw.parquet", index=False)


#safe check: check data are correctly merged 
# Ensure all timestamps are datetime
btc_filtered["timestamp"] = pd.to_datetime(btc_filtered["timestamp"])
trend_filtered["timestamp"] = pd.to_datetime(trend_filtered["timestamp"])
fear_filled["timestamp"] = pd.to_datetime(fear_filled["timestamp"])
final_df["timestamp"] = pd.to_datetime(final_df["timestamp"])

# Get common dates and pick 3 randomly
common_dates = set(btc_filtered["timestamp"]) & set(trend_filtered["timestamp"]) & set(fear_filled["timestamp"])
sample_dates = random.sample(list(common_dates), 3)

# Loop through dates and compare
for date in sample_dates:
    print(f"\n--- 🔎 Checking {date.date()} ---")
    
    btc_row = btc_filtered[btc_filtered["timestamp"] == date].reset_index(drop=True)
    trend_row = trend_filtered[trend_filtered["timestamp"] == date].reset_index(drop=True)
    fear_row = fear_filled[fear_filled["timestamp"] == date].reset_index(drop=True)
    final_row = final_df[final_df["timestamp"] == date].reset_index(drop=True)

    # Check BTC
    btc_match = btc_row.equals(final_row[btc_row.columns])
    print("BTC check:", "✅ Match" if btc_match else "❌ Mismatch")
    
    # Check Trend
    trend_match = trend_row.equals(final_row[trend_row.columns])
    print("Google Trends check:", "✅ Match" if trend_match else "❌ Mismatch")
    
    # Check Fear
    fear_match = fear_row.equals(final_row[fear_row.columns])
    print("Fear & Greed Index check:", "✅ Match" if fear_match else "❌ Mismatch")
