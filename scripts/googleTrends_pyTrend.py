#this function fetches google trends data directly from google trends website through pyTrend

from pytrends.request import TrendReq
import pandas as pd
from datetime import datetime
import time

def fetch_daily_trends(keyword="Bitcoin", start_year=2017, end_year=2024, sleep_time=30):
    pytrends = TrendReq()
    all_data = []

    for year in range(start_year, end_year + 1):
        intervals = [
            (f"{year}-01-01", f"{year}-06-30"),
            (f"{year}-07-01", f"{year}-12-31")
        ]

        for start_date, end_date in intervals:
            try:
                pytrends.build_payload([keyword], timeframe=f"{start_date} {end_date}")
                df = pytrends.interest_over_time()
                if not df.empty:
                    df = df.rename(columns={keyword: "google_trends"})
                    df = df.drop(columns=["isPartial"])
                    all_data.append(df)
                    print(f"✅ Retrieved data: {start_date} to {end_date}")
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    print(f"❌ Error 429 (Too Many Requests) detected for {start_date} to {end_date}. Stopping execution.")
                    return pd.DataFrame()  # or raise SystemExit if you want to halt the script
                else:
                    print(f"⚠️ Error retrieving data from {start_date} to {end_date}: {e}")
                    time.sleep(10)
                    continue

            time.sleep(sleep_time)

    if all_data:
        full_df = pd.concat(all_data)
        full_df = full_df[~full_df.index.duplicated(keep='first')]
        full_df = full_df.sort_index()
        return full_df
    else:
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    keyword = "Bitcoin"
    start_year = 2017
    end_year = 2024

    df_daily = fetch_daily_trends(keyword, start_year, end_year)
    if not df_daily.empty:
        df_daily.to_csv(f"data/google_trends_{keyword.lower()}_daily_{start_year}_{end_year}.csv")
        print("✅ Daily Google Trends data saved.")
    else:
        print("⚠️ No data fetched or script halted due to error.")
