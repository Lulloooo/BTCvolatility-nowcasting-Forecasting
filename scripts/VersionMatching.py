## Scalers scikit-learn version matching
# be sure to run it in the virtual enviroment named VenvScaling

#import needed packages
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
import joblib
import os
#get the current directory to reference files
print(os.getcwd())

#load the needed files
#save the x
X_tr = pd.read_csv("data/unscaled_data/X_tr_ewm.csv", index_col=0)
X_val = pd.read_csv("data/unscaled_data/X_val_ewm.csv", index_col=0)
X_tt = pd.read_csv("data/unscaled_data/X_tt_ewm.csv", index_col=0)
#save the y
y_tr = pd.read_csv("data/unscaled_data/y_tr_ewm.csv", index_col=0)
y_val = pd.read_csv("data/unscaled_data/y_val_ewm.csv", index_col=0)
y_tt = pd.read_csv("data/unscaled_data/y_tt_ewm.csv", index_col=0)


#scale the data
# Scaling all 3 sets and keeping boolean unchanged
num_cols = X_tr.select_dtypes(include=["int64", "float64"]).columns
# Initialize StandardScaler
scaler = StandardScaler()
# Create copies to keep original structure
X_tr_scal = X_tr.copy()
X_val_scal = X_val.copy()
X_tt_scal = X_tt.copy()
# Fit scaler on training data & transform all sets
X_tr_scal[num_cols] = scaler.fit_transform(X_tr[num_cols])  # Fit + transform on training
X_val_scal[num_cols] = scaler.transform(X_val[num_cols])    # Transform validation
X_tt_scal[num_cols] = scaler.transform(X_tt[num_cols])      # Transform test
# Display the transformed dataframes
print("Scaled Training dataset:")
print(X_tr_scal.head())
print()
print("Scaled Validation dataset:")
print(X_val_scal.head())
print()
print("Scaled testing dataset:")
print(X_tt_scal.head())

# Fit scaler only on training set
scaler_y = StandardScaler()
y_tr_scal = pd.DataFrame(scaler_y.fit_transform(y_tr.values.reshape(-1, 1)), columns=['target'])
# Then transform validation and test using the same scaler
y_val_scal = pd.DataFrame(scaler_y.transform(y_val.values.reshape(-1, 1)), columns=['target'])
y_tt_scal = pd.DataFrame(scaler_y.transform(y_tt.values.reshape(-1, 1)), columns=['target'])
# Display the transformed dataframes
print("Scaled Training dataset:")
print(y_tr_scal.head())
print()
print("Scaled Validation dataset:")
print(y_val_scal.head())
print()
print("Scaled testing dataset:")
print(y_tt_scal.head())

#save the scalers
joblib.dump(scaler, 'data/scaler_x_aws.pkl')
joblib.dump(scaler_y, 'data/scaler_y_aws.pkl')