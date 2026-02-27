################### PACKAGES
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
from pytrends.request import TrendReq
import joblib
import gradio as gr
from sklearn.preprocessing import StandardScaler


################### MODEL LOADING
# Load XGBoost and NGBoost
xgb_artifact = joblib.load("models/xgb_volatility_model_updated.joblib")
xgb_model = xgb_artifact["model"]
xgb_features = xgb_artifact["feature_names"]

ngb_artifact = joblib.load("models/ngb_volatility_model_updated.joblib")
ngb_model = ngb_artifact["model"]
ngb_features = ngb_artifact["feature_names"]

forecast_ngb_artifact = joblib.load("models/Forecast_ngb_volatility_model_updated.joblib")
forecast_ngb_model = forecast_ngb_artifact["model"]
forecast_ngb_features = forecast_ngb_artifact["feature_names"]

# Load KMeans + scaler
kmeans_artifact = joblib.load("models/kmeans_model_updated.joblib")
kmeans_model = kmeans_artifact["model"]
cluster_scaler = kmeans_artifact["scaler"]
# Only use the features that were actually used during training
#cluster_features_names = ["close", "volume", "trend", "fg_index"]
cluster_features_names = kmeans_artifact["feature_names"]

################### FEATURE GATHERING
def fetch_ohlcv_last_n_days(date="2026-01-18", n_days=90):
    end = pd.to_datetime(date)
    start = end - timedelta(days=n_days)

    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    params = {
        "vs_currency": "usd",
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

    df = df.resample("1D").agg(
        open=("close", "first"),
        high=("close", "max"),
        low=("close", "min"),
        close=("close", "last"),
        volume=("volume", "sum")
    ).dropna()

    return df

def fetch_fg_last_6_days(df):
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    response = requests.get(url)
    fg_df = pd.DataFrame(response.json()["data"])
    fg_df["timestamp"] = pd.to_datetime(fg_df["timestamp"], unit="s")
    fg_df.set_index("timestamp", inplace=True)
    fg_df = fg_df[["value"]].astype(float)
    fg_df.rename(columns={"value":"fg_index"}, inplace=True)
    fg_df = fg_df.reindex(df.index, method="ffill")
    df["fg_index"] = fg_df["fg_index"].values
    return df

def fetch_google_trend_for_date(keyword="Bitcoin", target_date="2024-01-15", window=7):
    pytrends = TrendReq(hl="en-US", tz=360)
    target_date = pd.to_datetime(target_date)
    start_date = (target_date - timedelta(days=window)).strftime("%Y-%m-%d")
    end_date = target_date.strftime("%Y-%m-%d")
    timeframe = f"{start_date} {end_date}"
    try:
        pytrends.build_payload([keyword], timeframe=timeframe)
        df = pytrends.interest_over_time()
        if df.empty:
            return 0
        df = df.rename(columns={keyword: "trend"})
        df = df.drop(columns=["isPartial"], errors="ignore")
        df.index = pd.to_datetime(df.index)
        if target_date in df.index:
            return df.loc[target_date, "trend"]
        else:
            return 0
    except:
        return 0


################### FEATURE ENGINEERING
def engineer_features(df):
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["hl_spread"] = df["high"] - df["low"]
    df["co_spread"] = df["close"] - df["open"]
    df["momentum_3"] = df["close"] - df["close"].shift(3)
    df["vol_change"] = df["volume"] - df["volume"].shift(1)
    df["rolling_std_5"] = df["log_return"].rolling(5).std()
    # Fill NaNs for first few rows
    df = df.fillna(0)
    return df

################### ASSIGN CLUSTER
def assign_cluster(df):
    cluster_features = df[cluster_features_names]
    scaled = cluster_scaler.transform(cluster_features)
    df["cluster"] = kmeans_model.predict(scaled)
    return df

################### PREDICTION FUNCTION
def predict_volatility(date):
    # Fetch raw data
    df = fetch_ohlcv_last_n_days(date=date, n_days=90)
    df = fetch_fg_last_6_days(df)
    trend_value = fetch_google_trend_for_date("Bitcoin", date)
    df["trend"] = trend_value
    df["trend"] = df["trend"].ffill().fillna(0)

    # Feature engineering
    df = engineer_features(df)

    # Clustering
    df = assign_cluster(df)

    assert "cluster" in df.columns, "Cluster feature missing after assignment"


    # ---------------- NOWCASTING ----------------
    X_xgb = df.iloc[-1:][xgb_features]
    X_ngb = df.iloc[-1:][ngb_features]

    point = xgb_model.predict(X_xgb)[0]
    dist = ngb_model.pred_dist(X_ngb)
    low, high = dist.ppf([0.025, 0.975])

    # ---------------- FORECASTING (t+1) ----------------
    X_ngb_fore = df.iloc[-1:][forecast_ngb_features]

    for_point = forecast_ngb_model.predict(X_ngb_fore)[0]
    for_dist = forecast_ngb_model.pred_dist(X_ngb_fore)
    for_low, for_high = for_dist.ppf([0.025, 0.975])

    return point, low, high, for_point, for_low, for_high



################### GRADIO INTERFACE

################### HELPER FUNCTION TO RETURN TABLE ###################
def predict_volatility_for_table(date):
    """
    Returns a DataFrame with Nowcast and Forecast predictions for the given date.
    """
    # Run your existing predict_volatility function
    point, low, high, for_point, for_low, for_high = predict_volatility(date)

    # Create a DataFrame to display nicely
    data = {
        "Type": ["Nowcast (t)", "Forecast (t+1)"],
        "Volatility": [point, for_point],
        "Low 95% CI": [low, for_low],
        "High 95% CI": [high, for_high]
    }

    df = pd.DataFrame(data)

    # Round values for better readability
    df[["Volatility", "Low 95% CI", "High 95% CI"]] = df[["Volatility", "Low 95% CI", "High 95% CI"]].round(4)

    return df

################### GRADIO WRAPPER ###################
def gradio_predict(date):
    """
    Wrapper for Gradio. Returns a DataFrame for display.
    """
    # Validate date format
    try:
        pd.to_datetime(date)
    except:
        return pd.DataFrame({"Error": ["❌ Invalid date format. Use YYYY-MM-DD."]})

    # Attempt to predict
    try:
        df = predict_volatility_for_table(date)
        return df
    except Exception as e:
        return pd.DataFrame({"Error": [f"⚠️ Error while computing prediction:\n{str(e)}"]})

################### GRADIO INTERFACE ###################
demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(label="📆 Date (YYYY-MM-DD)"),
    outputs=gr.Dataframe(label="Volatility Predictions", headers=["Type", "🎯 Volatility", "Low 95% CI", "High 95% CI"]),
    title="BTC Volatility Predictor",
    description=(
        "Enter a date to get predicted BTC volatility and 95% confidence intervals.\n"
        "Nowcast = today's volatility, Forecast = next day's volatility."
    )
)

if __name__ == "__main__":
    demo.launch()