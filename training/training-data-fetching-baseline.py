################
# FETCH & ENGINEER UP-TO-DATE DATA FOR PERIODIC AUTOMATIC RE-TRAINING
# This parse OHLCV data (from coingeko), Fear and greed index and daily google trend data
# from df_upTodate.csv last date to today.
# Then it re-train the k-means cluster alrogith with the new data and assign new clusters
# finally, it engineers the features used for training along with the target (ewma volatility)
# In the end, it saves both x_training and y_training
#### OUTPUT: df_upTodate.csv, k_means_model_updated.joblib, x_training, y_training,
#### UPDATED MODEL(S): k_means_model_updated.joblib (clustering)
################

#PACKAGES
from pathlib import Path
import pandas as pd
import requests
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import time
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

############ PATHS
base_dir = Path(__file__).resolve().parents[1]   # BTC-volatility-forecasting/
data_dir = base_dir / "data"/ "training_data"

############ FUNCTIONS
##### DATA FETCHING
### OHLCV
def update_ohlcv_coingecko(symbol = "bitcoin", vs_currency = "usd"):
    #define start and end date
    start = pd.Timestamp(extended_start)
    end = datetime.utcnow()
    
    #if start is > end halt it
    if start >= end:
        print("🟢 OHLCV already up to date")
        return pd.DataFrame()
    #else pharse the last dates
    else:
        #define the endpoint
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart/range"
        params = {
            "vs_currency": vs_currency,
            "from": int(start.timestamp()),
            "to": int(end.timestamp())
        }
        #make the request
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        #get prices & volumne
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
        volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
        ohlcv = prices.merge(volumes, on="timestamp")
        #set index & timestamp
        ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], unit="ms")
        ohlcv.set_index("timestamp", inplace=True)
        #normalize timezone
        ohlcv.index = pd.to_datetime(ohlcv.index).tz_localize(None).normalize()
        # OHLC resample
        ohlcv = ohlcv.resample("1D").agg(
            open=("close", "first"),
            high=("close", "max"),
            low=("close", "min"),
            close=("close", "last"),
            volume=("volume", "sum")
        ).dropna()
    return ohlcv
### FG INDEX
def fear_greed_index():
    #API to get all historical data in JSON
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    #make an HTTP get req
    response = requests.get(url)
    #convert http in json and extract "data"
    data = response.json()["data"]
    #convert data (a list of dictionaries) into a df 
    fg_index = pd.DataFrame(data)
    #change timestamp to datetime format in seconds
    fg_index["timestamp"] = pd.to_datetime(fg_index["timestamp"], unit = "s")
    #set timestamp as the index
    fg_index.set_index("timestamp", inplace = True)
    #normalize it
    fg_index.index = pd.to_datetime(fg_index.index).tz_localize(None).normalize()
    #select only the value column and change it into float
    fg_index = fg_index[["value"]].astype(float)
    #rename the column
    fg_index.rename(columns = {"value" : "fg_index",
                               "timestamp" : "date"},
                    inplace = True)
    #sort the index
    fg_index = fg_index.sort_index()
    #define end date as today
    end = datetime.utcnow().date()
    #define start date
    start = extended_start
    #slice time range
    fg_index = fg_index.loc[start:end]
    
    #return the final df
    return fg_index
### GOOGLE TRENDS
def fetch_daily_trends(keyword="Bitcoin", start_date=None, sleep_time=5):

    if start_date is None:
        start_date = extended_start

    start_date = pd.to_datetime(start_date)
    end_date = pd.Timestamp.today().normalize()

    pytrends = TrendReq()

    all_data = []

    current_start = start_date

    while current_start < end_date:

        current_end = min(current_start + timedelta(days=90), end_date)

        timeframe = f"{current_start.strftime('%Y-%m-%d')} {current_end.strftime('%Y-%m-%d')}"
        print("Fetching:", timeframe)

        try:
            pytrends.build_payload([keyword], timeframe=timeframe)
            df = pytrends.interest_over_time()

            if not df.empty:
                df = df.rename(columns={keyword: "trend"})
                df = df.drop(columns=["isPartial"], errors="ignore")
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                all_data.append(df)

        except Exception as e:
            print("Error:", e)

        time.sleep(sleep_time)
        current_start = current_end + timedelta(days=1)

    if not all_data:
        return pd.DataFrame()

    trend = pd.concat(all_data)
    trend = trend[~trend.index.duplicated(keep="last")]
    trend = trend.sort_index()

    return trend


##### DATA TRANSFORMATION
### RE-SCALING & CLUSTERING
def scaling_and_clustering(df, n_clusters=8):
    #keep variables used for clustering
    clusterdf = df.drop(columns=["high", "low", "open"])
    #scale data
    scaler = StandardScaler()
    clusterScal = scaler.fit_transform(clusterdf)
    #fit the clustering (8-ks, as in production phase)
    #instantiate a 8-k model
    kmeans8 = KMeans(n_clusters=n_clusters, 
                     random_state=42)
    #fit the model
    kmeans8.fit(clusterScal)
    #add the cluster to the df
    #add the clusters to the original df
    df["cluster"] = kmeans8.labels_
    #save the artifacts
    kmean_artifact = {
    'model': kmeans8,
    'feature_names': clusterdf.columns.tolist(),
    'scaler': scaler
    }
    joblib.dump(
        kmean_artifact,
        base_dir / "models"/ "kmeans_model_updated.joblib")
    #return the df w/ clustering
    return df
### FEATURE ENGINEERING
def engineer_features(df):
    #log return
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    #high-low spread
    df["hl_spread"] = df["high"] - df["low"]
    #difference between current close and close 3 days ago
    df["momentum_3"] = df["close"] - df["close"].shift(3)
    #chane in trading volumes repsect to yesterday
    df["vol_change"] = df["volume"] - df["volume"].shift(1)
    #log return std over a 5-days rolling windows
    df["rolling_std_5"] = df["log_return"].rolling(5).std()
    # drop NaNs for first few rows
    df = df.dropna()
    #drop open, low and high (not useful variables)
    df = df.drop(columns=["open", "low", "high"])
    #return the df
    return df
### TARGET
#compute volatility with ewm technique
def compute_volatility(df):
    #compute volatility as ewma with a 5-days span
    df['volatility'] = df['log_return'].ewm (span=5).std()
    #drop NaNs
    df = df.dropna(how="any")
    #define the target variable (y)
    y_training = df['volatility']
    #define the regressors df
    x_training = df.copy()
    #drop volatility (y) from regressors
    x_training = x_training.drop(['volatility'], axis = 1)
    #return y and Xs
    return x_training, y_training
    

############ EXECUTIONS
##### BASELINE DF LOADING
##load not-enginerred df
X_start = pd.read_csv(base_dir / "data"/ "crypto_volatility_clean.csv", 
                      parse_dates = ["date"])
#sort by date
X_start = X_start.sort_values("date")
#store last date
last_date = X_start["date"].max()
#debugging
if X_start is not None:
    print("Regressors loaded") 
#show last date
print(f"last date: {last_date}")
#define start date
start = last_date + timedelta(days=1)
#define and extended date for feature enginnering (pharse 5 days before, as the biggest loopback
# is 5 days for volatility) Note. +2 guarantees minimum required data thanks to extra fetching
lookback = 5
extended_start = start - timedelta(days=lookback+2)
# NOTE. we can assume that 5 days are 5 observations without missing entries,
# as crypto trades everyday (that's not the case with stocks)

##### NEW DATA GATHERING & MERGING
#ohlcv
ohlcv = update_ohlcv_coingecko()
#fg_index 
fg_index = fear_greed_index()
#google trend
trend = fetch_daily_trends()
#align
trend = trend.reindex(ohlcv.index, method="ffill")
##### MERGING
# Merge fg index and trend on index (date)
index_and_trends = fg_index.join(trend, how="outer")
#Add ohlcv (on index, i.e date)
df_updated = ohlcv.join(index_and_trends, how="outer")
#rename google trend column to trend
df_updated = df_updated.rename(columns={"google_trends": "trend"})

##### BASELINE DATASET UPDATING (X_start + last month data)
#define the base df
X_base = X_start.set_index("date")
#add the new rows
X_full = X_base.combine_first(df_updated)
#show it
print("df_upTodate preview:")
print(trend.tail())
print(X_full.tail(10))
#define column order as in baseline
column_order = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "fg_index",
    "trend"
]
#change the order in the new df
X_full = X_full[column_order]
#Save the updated datase
X_full.to_csv(data_dir/ "df_upToDate.csv", index_label= "date")


##### DATASET ENGINEERING
#clustering (assign cluster)
df_cluster = scaling_and_clustering(X_full)
#debugging
#print(df_cluster.head(10))
#feature engineering
df_fe = engineer_features(df_cluster)
#debugging
#print(df_fe.head(10))
#drop first day


##### TARGET & REGRESSOR SPLIT
#target - regressor split
x_training, y_training = compute_volatility(df_fe)
#store both regressors and target
x_training.to_csv(data_dir/ "x_training.csv")
y_training.to_csv(data_dir/ "y_training.csv")

