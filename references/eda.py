# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import StackingRegressor

# Load the datasets
balancing_data = pd.read_csv('/Users/sravansridhar/Desktop/Edp_datathon/data/balancing_data.csv')
demand_load_data = pd.read_csv('/Users/sravansridhar/Desktop/Edp_datathon/data/demand_load_data.csv')
generation_data = pd.read_csv('/Users/sravansridhar/Desktop/Edp_datathon/data/generation_data.csv')
price_data = pd.read_csv('/Users/sravansridhar/Desktop/Edp_datathon/data/price_data.csv')

# Convert GMT Time to datetime
for df in [balancing_data, demand_load_data, generation_data, price_data]:
    df['GMT Time'] = pd.to_datetime(df['GMT Time'])

# Replace 'No Data Available' with NaN
for df in [balancing_data, demand_load_data, generation_data, price_data]:
    df.replace('No Data Available', np.nan, inplace=True)

# Merge datasets on GMT Time
merged_data = pd.merge(balancing_data, demand_load_data, on='GMT Time')
merged_data = pd.merge(merged_data, generation_data, on='GMT Time')
merged_data = pd.merge(merged_data, price_data, on='GMT Time')

# Convert all columns to numeric
for col in merged_data.columns:
    if merged_data[col].dtype == 'object':
        merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

# Handle missing values
merged_data.fillna(method='ffill', inplace=True)
merged_data.fillna(method='bfill', inplace=True)

# Feature Engineering
merged_data['Hour'] = merged_data['GMT Time'].dt.hour
merged_data['Day'] = merged_data['GMT Time'].dt.day
merged_data['Month'] = merged_data['GMT Time'].dt.month
merged_data['Year'] = merged_data['GMT Time'].dt.year
merged_data['DayOfWeek'] = merged_data['GMT Time'].dt.dayofweek

# Lag features
for lag in range(1, 25):
    merged_data[f'System_Price_Lag_{lag}'] = merged_data['System Price (ESO Outturn) - GB (£/MWh)'].shift(lag)
    merged_data[f'NIV_Lag_{lag}'] = merged_data['NIV Outturn (+ve long) - GB (MW)'].shift(lag)

# Rolling statistics
merged_data['Rolling_Mean_24'] = merged_data['System Price (ESO Outturn) - GB (£/MWh)'].rolling(window=24).mean()
merged_data['Rolling_Std_24'] = merged_data['System Price (ESO Outturn) - GB (£/MWh)'].rolling(window=24).std()

# Drop rows with NaN values created by lag features
merged_data.dropna(inplace=True)

# Splitting the data into training and testing sets
X = merged_data.drop(columns=['GMT Time', 'System Price (ESO Outturn) - GB (£/MWh)', 'NIV Outturn (+ve long) - GB (MW)'])
y_price = merged_data['System Price (ESO Outturn) - GB (£/MWh)']
y_niv = merged_data['NIV Outturn (+ve long) - GB (MW)']

X_train, X_test, y_price_train, y_price_test = train_test_split(X, y_price, test_size=0.2, random_state=42)
X_train, X_test, y_niv_train, y_niv_test = train_test_split(X, y_niv, test_size=0.2, random_state=42)

# Define the base models
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=100, random_state=42, tree_method='gpu_hist', predictor='gpu_predictor')),
    ('lgbm', LGBMRegressor(n_estimators=100, random_state=42, device='gpu'))
]

# Define the stacking regressor
stacking_regressor = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression()
)

# Define the voting regressor
voting_regressor = VotingRegressor(estimators=base_models)

# Define the bagging regressor
bagging_regressor = BaggingRegressor(estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_estimators=10, random_state=42)

# Train and evaluate the models
models = {
    'Stacking Regressor': stacking_regressor,
    'Voting Regressor': voting_regressor,
    'Bagging Regressor': bagging_regressor
}

for name, model in models.items():
    model.fit(X_train, y_price_train)
    y_price_pred = model.predict(X_test)
    rmse_price = mean_squared_error(y_price_test, y_price_pred, squared=False)
    print(f'{name} RMSE for System Price: {rmse_price:.4f}')
    
    model.fit(X_train, y_niv_train)
    y_niv_pred = model.predict(X_test)
    rmse_niv = mean_squared_error(y_niv_test, y_niv_pred, squared=False)
    print(f'{name} RMSE for Net Imbalance Volume: {rmse_niv:.4f}')

# Generating predictions for October 1, 2024
october_1_data = merged_data[merged_data['GMT Time'].dt.date == pd.to_datetime('2024-10-01').date()]
october_1_predictions_price = stacking_regressor.predict(october_1_data.drop(columns=['GMT Time', 'System Price (ESO Outturn) - GB (£/MWh)', 'NIV Outturn (+ve long) - GB (MW)']))
october_1_predictions_niv = stacking_regressor.predict(october_1_data.drop(columns=['GMT Time', 'System Price (ESO Outturn) - GB (£/MWh)', 'NIV Outturn (+ve long) - GB (MW)']))

# Creating the output file
output = pd.DataFrame({
    'GMT_TIME': october_1_data['GMT Time'],
    'SYSTEM_PRICE': october_1_predictions_price,
    'NIV_OUTTURN': october_1_predictions_niv
})

output.to_csv('predictions_october_1_2024.csv', index=False)
print(output.head())