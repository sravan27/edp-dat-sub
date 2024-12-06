# Import necessary libraries
import pandas as pd
import numpy as np
import re
import joblib
import multiprocessing
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
import os

# Function to normalize column names
def normalize_column_names(df):
    df = df.copy()
    # Replace non-alphanumeric characters with underscores
    df.columns = df.columns.str.replace(r"[^\w\s]", "_", regex=True)
    # Replace any whitespace with a single underscore
    df.columns = df.columns.str.replace(r"\s+", "_", regex=True)
    # Convert to lowercase
    df.columns = df.columns.str.lower()
    # Replace multiple underscores with a single underscore
    df.columns = df.columns.str.replace(r"_+", "_", regex=True)
    # Remove leading and trailing underscores
    df.columns = df.columns.str.strip('_')
    return df

# Function to load and preprocess a single dataset
def load_and_preprocess_dataset(filepath):
    df = pd.read_csv(filepath)
    df.replace('No Data Available', np.nan, inplace=True)
    df = normalize_column_names(df)
    if 'gmt_time' in df.columns:
        df['gmt_time'] = pd.to_datetime(df['gmt_time'])
    else:
        raise KeyError(f"The required column 'gmt_time' is missing in the dataset {filepath}.")
    # Convert all columns except 'gmt_time' to numeric
    cols = df.columns.drop('gmt_time')
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    return df

# Load datasets
print("Loading and preprocessing datasets...")
balancing_data = load_and_preprocess_dataset('balancing_data.csv')
demand_load_data = load_and_preprocess_dataset('demand_load_data.csv')
generation_data = load_and_preprocess_dataset('generation_data.csv')
price_data = load_and_preprocess_dataset('price_data.csv')

# Merge datasets on 'gmt_time'
print("Merging datasets...")
data_frames = [balancing_data, demand_load_data, generation_data, price_data]
merged_data = data_frames[0]
for df in data_frames[1:]:
    merged_data = pd.merge(merged_data, df, on='gmt_time', how='outer')

# Handle missing values
print("Handling missing values...")
merged_data.sort_values('gmt_time', inplace=True)
merged_data.fillna(method='ffill', inplace=True)
merged_data.fillna(method='bfill', inplace=True)
merged_data.dropna(inplace=True)

# Ensure all data types are numeric except 'gmt_time'
print("Ensuring data types are correct...")
cols = merged_data.columns.drop('gmt_time')
merged_data[cols] = merged_data[cols].apply(pd.to_numeric, errors='coerce')

# Feature engineering
print("Feature engineering...")
# Add time-based features
merged_data['hour'] = merged_data['gmt_time'].dt.hour
merged_data['day_of_week'] = merged_data['gmt_time'].dt.dayofweek
merged_data['month'] = merged_data['gmt_time'].dt.month
merged_data['is_weekend'] = (merged_data['day_of_week'] >= 5).astype(int)

# Add lag features
print("Adding lag features...")
lag_features = ['system_price_eso_outturn_gb_mwh', 'niv_outturn_ve_long_gb_mw']
for lag in [1, 24, 48]:  # Lag by 1 hour, 1 day, 2 days
    for col in lag_features:
        if col in merged_data.columns:
            merged_data[f'{col}_lag_{lag}'] = merged_data[col].shift(lag)
        else:
            print(f"Warning: Column {col} not found in merged_data.")

# Add rolling statistics
print("Adding rolling statistics...")
for window in [7, 30]:  # Weekly and monthly
    for col in lag_features:
        if col in merged_data.columns:
            merged_data[f'{col}_rolling_mean_{window}'] = merged_data[col].rolling(window).mean()
        else:
            print(f"Warning: Column {col} not found in merged_data.")

# Drop rows with NaN values created due to lags/rolling
merged_data.dropna(inplace=True)

# Define features and targets
print("Defining features and targets...")
X = merged_data.drop(columns=['gmt_time', 'system_price_eso_outturn_gb_mwh', 'niv_outturn_ve_long_gb_mw'])
y_price = merged_data['system_price_eso_outturn_gb_mwh']
y_niv = merged_data['niv_outturn_ve_long_gb_mw']

# Ensure there are no remaining non-numeric columns
print("Verifying that all features are numeric...")
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print(f"Converting non-numeric columns to numeric: {non_numeric_cols}")
    X[non_numeric_cols] = X[non_numeric_cols].apply(pd.to_numeric, errors='coerce')
    X.dropna(inplace=True)
    y_price = y_price.loc[X.index]
    y_niv = y_niv.loc[X.index]

# Implement a pipeline with scaling
print("Creating pipelines...")
from sklearn.preprocessing import StandardScaler

# Define the preprocessing and modeling pipeline
def create_pipeline(model):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    return pipeline

# Define base models
print("Defining base models...")
# XGBoost parameters
xgb_params = {
    'random_state': 42,
    'objective': 'reg:squarederror',
    'n_jobs': multiprocessing.cpu_count()
}

# LightGBM parameters
lgbm_params = {
    'random_state': 42,
    'n_jobs': multiprocessing.cpu_count()
}

base_models = [
    ('rf', RandomForestRegressor(random_state=42, n_jobs=multiprocessing.cpu_count())),
    ('gb', GradientBoostingRegressor(random_state=42)),
    ('xgb', XGBRegressor(**xgb_params)),
    ('lgbm', LGBMRegressor(**lgbm_params))
]

# Create pipelines for base models
base_pipelines = [(name, create_pipeline(model)) for name, model in base_models]

# Define stacking regressors
print("Defining stacking regressors...")
stacking_regressor_price = StackingRegressor(
    estimators=base_pipelines,
    final_estimator=Ridge(),
    passthrough=True,
    n_jobs=multiprocessing.cpu_count()
)
stacking_regressor_niv = StackingRegressor(
    estimators=base_pipelines,
    final_estimator=Ridge(),
    passthrough=True,
    n_jobs=multiprocessing.cpu_count()
)

# Hyperparameter tuning using RandomizedSearchCV
print("Starting hyperparameter tuning for System Price model...")
param_distributions = {
    'rf__model__n_estimators': [100, 200],
    'rf__model__max_depth': [None, 10],
    'gb__model__n_estimators': [100, 200],
    'gb__model__learning_rate': [0.05, 0.1],
    'xgb__model__n_estimators': [100, 200],
    'xgb__model__learning_rate': [0.05, 0.1],
    'lgbm__model__n_estimators': [100, 200],
    'lgbm__model__learning_rate': [0.05, 0.1],
    'final_estimator__alpha': [0.1, 1.0, 10.0]
}

def perform_random_search(model, X, y, param_distributions, model_name):
    tscv = TimeSeriesSplit(n_splits=3)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=20,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        verbose=2,
        n_jobs=multiprocessing.cpu_count(),
        random_state=42
    )
    random_search.fit(X, y)
    best_model = random_search.best_estimator_
    print(f"Best parameters for {model_name}: {random_search.best_params_}")
    # Save the best model
    joblib.dump(best_model, f'best_model_{model_name}.pkl')
    return best_model

# Perform hyperparameter tuning for System Price
best_model_price = perform_random_search(
    stacking_regressor_price,
    X,
    y_price,
    param_distributions,
    'price'
)

# Perform hyperparameter tuning for NIV
print("Starting hyperparameter tuning for NIV model...")
best_model_niv = perform_random_search(
    stacking_regressor_niv,
    X,
    y_niv,
    param_distributions,
    'niv'
)

# Save the scalers and pipelines for the Streamlit app
print("Saving scalers and pipelines for Streamlit app...")
joblib.dump(best_model_price, 'pipeline_model_price.pkl')
joblib.dump(best_model_niv, 'pipeline_model_niv.pkl')

# Prepare features for October 1, 2024
print("Preparing features for October 1, 2024...")
# Generate date range for October 1, 2024
dates_oct1 = pd.date_range(start='2024-10-01 00:00:00', end='2024-10-01 23:30:00', freq='30T')
october_1_features = pd.DataFrame({'gmt_time': dates_oct1})

# Create the same features as in training
print("Creating features for October 1, 2024...")
october_1_features['hour'] = october_1_features['gmt_time'].dt.hour
october_1_features['day_of_week'] = october_1_features['gmt_time'].dt.dayofweek
october_1_features['month'] = october_1_features['gmt_time'].dt.month
october_1_features['is_weekend'] = (october_1_features['day_of_week'] >= 5).astype(int)

# Estimate lag and rolling features using historical data
print("Estimating lag and rolling features using historical data...")
# Calculate historical averages for lag and rolling features
historical_data = merged_data.copy()
historical_data['time'] = historical_data['gmt_time'].dt.time
october_1_features['time'] = october_1_features['gmt_time'].dt.time

# List of features to estimate
features_to_estimate = X.columns.tolist()

# Calculate mean values by time for features
mean_features = historical_data.groupby('time')[features_to_estimate].mean().reset_index()

# Merge mean features with october_1_features
october_1_features = pd.merge(october_1_features, mean_features, on='time', how='left')

# Drop unnecessary columns
october_1_features.drop(columns=['time', 'gmt_time'], inplace=True)

# Ensure no missing values in features
print("Checking for missing values in October 1 features...")
if october_1_features.isnull().values.any():
    print("Filling missing values with historical means...")
    october_1_features.fillna(october_1_features.mean(), inplace=True)

# Predict for October 1, 2024
print("Predicting for October 1, 2024...")
october_1_predictions_price = best_model_price.predict(october_1_features)
october_1_predictions_niv = best_model_niv.predict(october_1_features)

# Create output DataFrame
output = pd.DataFrame({
    'GTM_TIME': dates_oct1,
    'SYSTEM_PRICE': october_1_predictions_price,
    'NIV_OUTTURN': october_1_predictions_niv
})

# Save predictions to CSV
print("Saving predictions...")
output.to_csv('predictions_october_1_2024.csv', index=False)
print("Predictions saved to 'predictions_october_1_2024.csv'.")

# Evaluate model performance on the training set (optional)
print("Evaluating model performance on training data...")
y_pred_price = best_model_price.predict(X)
rmse_price = mean_squared_error(y_price, y_pred_price, squared=False)
print(f"Training RMSE for System Price: {rmse_price:.4f}")

y_pred_niv = best_model_niv.predict(X)
rmse_niv = mean_squared_error(y_niv, y_pred_niv, squared=False)
print(f"Training RMSE for NIV: {rmse_niv:.4f}")

# Save the merged data and features for Streamlit app
print("Saving processed data for Streamlit app...")
merged_data.to_csv('merged_data_processed.csv', index=False)
X.to_csv('features.csv', index=False)
y_price.to_csv('target_price.csv', index=False)
y_niv.to_csv('target_niv.csv', index=False)
print("Data saved.")
