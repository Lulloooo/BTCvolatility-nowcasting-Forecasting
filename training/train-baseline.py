################
# DEPLOYED MODEL MONTHLY RE-TRAINING & UPDATING
# This loads deployed models' last version, re-train them based on 
#the new data fetched (thanks to training-data-fetching.py)
#and save the updated models.
#### OUTPUT: xgb_volatility_model_updated.joblib, ngb_volatility_model_updated.joblib, 
#            Forecast_ngb_volatility_model_updated.joblib
#### UPDATED MODEL(S): xgb_volatility_model_updated.joblib (point nowcasting), ngb_volatility_model_updated.joblib (interval nowcasting), 
#                      Forecast_ngb_volatility_model_updated.joblib (point + interval forecasting)
################

############ PACKAGES
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
from pathlib import Path
import joblib

# ML
from xgboost import XGBRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore

############ PATHS
#define the base directory
base_dir = Path(__file__).resolve().parents[1]
#updated models dir
model_dir = base_dir / "models"
#baseline models dir (first time only)
baseline_model_dir = base_dir / "models"/ "baseline_models"
#data dir
data_dir = base_dir / "data"/ "training_data"

############ DATA LOADING
#regressors
x = pd.read_csv(data_dir / "x_training.csv",
                index_col = 0,
                parse_dates = True)
print(x.head())
#target
y = pd.read_csv(data_dir / "y_training.csv",
                index_col = 0,
                parse_dates = True)
print(y.head())
#flatten y (as scikit learn expects)
y = y.values.ravel()

############ TRAINING PIPELINE
def training_and_updating():

    ###### NOWCASTING 
    ### MODEL LOADING
    #load XGBoost artifact & model & features
    xgb_artifact = joblib.load(baseline_model_dir/ "xgb_volatility_model.joblib")
    xgb_model = xgb_artifact["model"]
    xgb_features = xgb_artifact["feature_names"]
    xgb_params = xgb_artifact["params"]
    #load NGboost for interval Nowcasting
    ngb_artifact = joblib.load(baseline_model_dir/ "ngb_volatility_model.joblib")
    ngb_model = ngb_artifact["model"]
    ngb_features = ngb_artifact["feature_names"]
    ngb_params = ngb_artifact["params"]
    ### MODEL TRAINING
    #train xgboost
    print("🚀 Training Nowcgasting XGBoost...")
    #instantiate the model
    xgb_model = XGBRegressor(**xgb_params)
    #fit the model
    xgb_model.fit(x, y)
    #train ngboost
    print("🚀 Training Nowcasting NGBoost...")
    ngb_model = NGBRegressor(**ngb_params)
    ngb_model.fit(x, y)
    ### MODEL SAVING
    #save updated xgboost (nowcasting)
    joblib.dump(
        {
            "model": xgb_model,
            "feature_names": xgb_features,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "params": xgb_model.get_params()
        },
        model_dir / "xgb_volatility_model_updated.joblib"
    )
    #save updated Ngboost (nowcasting)
    joblib.dump(
        {
            "model": ngb_model,
            "feature_names": ngb_features,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "params": ngb_model.get_params()
        },
        model_dir / "ngb_volatility_model_updated.joblib"
    )

    ###### FORECASTING
    ### MODEL LOADING
    forecast_ngb_artifact = joblib.load(baseline_model_dir/ "Forecast_ngb_volatility_model.joblib")
    forecast_ngb_model = forecast_ngb_artifact["model"]
    forecast_ngb_features = forecast_ngb_artifact["feature_names"]
    forecast_ngb_params = forecast_ngb_artifact["params"]
    ### MODEL TRAINING
    print("🚀 Training Forecasting NGBoost...")
    forecast_ngb_model = NGBRegressor(**forecast_ngb_params)
    forecast_ngb_model.fit(x, y)
    ### MODEL SAVING
    joblib.dump(
        {
            "model": forecast_ngb_model,
            "feature_names": forecast_ngb_features,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "params": forecast_ngb_model.get_params()
        },
        model_dir / "Forecast_ngb_volatility_model_updated.joblib"
    )
    #print confirmation
    print("✅ Training complete. Updated artifacts saved.")

############ EXECUTION
if __name__ == "__main__":
    training_and_updating()