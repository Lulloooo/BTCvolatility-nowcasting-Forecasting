
################
# MERGE ALL THE SPLIT DATA (TRAINING - VALIDATION - TESTING) in a single df
# (X_baseline and y_baseline). This:
# a) shows all the data used for tuning in the Analysis notebook
# b) is the starting point (left untouched) to whom each month data are added to re-train the model
#OUTPUTS: x_baseline_fe, y_baseline_fe
################

#import packages
from pathlib import Path
import pandas as pd

#directories
base_dir = Path(__file__).resolve().parents[1]   # BTC-volatility-forecasting/
data_dir = base_dir / "data"/ "analysis-and-tuning-data"
out_data_dir = base_dir / "data"

#regressors
X_tr = pd.read_csv(data_dir/ "X_tr_ewm.csv",index_col="date",parse_dates=["date"])
X_val = pd.read_csv(data_dir/ "X_val_ewm.csv",index_col="date",parse_dates=["date"])
X_tt = pd.read_csv(data_dir/ "X_tt_ewm.csv",index_col="date",parse_dates=["date"])
#target
y_tr = pd.read_csv(data_dir/ "y_tr_ewm.csv")
y_val = pd.read_csv(data_dir/ "y_val_ewm.csv")
y_tt = pd.read_csv(data_dir/ "y_tt_ewm.csv")

#debugging
if X_tr is not None and X_val is not None and X_tt is not None:
    print("all regressors dfs were loaded.") 
if y_tr is not None and y_val is not None and y_tt is not None:
    print("all targets dfs were loaded.") 


#concat all the dfs
#regressors
X_full = pd.concat([X_tr, X_val, X_tt])
#remove duplicate dates if any
X_full = X_full[~X_full.index.duplicated(keep='first')]
#sort chronologically
X_full = X_full.sort_index()
#show it
print("X_full preview:")
print(X_full.head(10))
#targets (y)
y_full = pd.concat([y_tr, y_val, y_tt])
#remove duplicate dates if any
y_full = y_full[~y_full.index.duplicated(keep='first')]
#sort chronologically
y_full = y_full.sort_index()
#show it
print("y_full preview:")
print(y_full.head(10))

# Drop the unwanted "Unnamed: 0" column if it exists
if "Unnamed: 0" in X_full.columns:
    X_full = X_full.drop(columns="Unnamed: 0")

if "Unnamed: 0" in y_full.columns:
    y_full = y_full.drop(columns="Unnamed: 0")
#save it
# X keeps date as index
X_full.to_csv(out_data_dir / "training_data"/ "X_baseline_fe.csv", index=True)

# y has no date index, just values
y_full.to_csv(out_data_dir / "training_data"/ "y_baseline_fe.csv", index=True)



