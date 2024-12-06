# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
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

# Extract the target variable (System Price)
system_price = merged_data['System Price (ESO Outturn) - GB (£/MWh)']

# Plot the time series
plt.figure(figsize=(15, 8))
plt.plot(merged_data['GMT Time'], system_price)
plt.title('System Price Over Time')
plt.xlabel('Time')
plt.ylabel('System Price (£/MWh)')
plt.show()

# Perform Augmented Dickey-Fuller test to check for stationarity
result = adfuller(system_price)
print('ADF Statistic:', result[0])
print('p-value:', result[1])
for key, value in result[4].items():
    print('Critical Values:')
    print(f'   {key}, {value}')

# Plot ACF and PACF to determine p and q
plt.figure(figsize=(12, 6))
plot_acf(system_price, lags=50)
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(system_price, lags=50)
plt.show()

# Fit ARIMA model with determined p and q
p = 0  # Example value, determine based on PACF plot
d = 0  # No differencing needed as the series is stationary
q = 1  # Example value, determine based on ACF plot

arima_model = ARIMA(system_price, order=(p, d, q))
arima_result = arima_model.fit()

# Print model summary
print(arima_result.summary())

# Forecast future values
forecast_steps = 48  # Forecasting for 24 hours with half-hour intervals
forecast = arima_result.forecast(steps=forecast_steps)

# Plot the forecast
plt.figure(figsize=(15, 8))
plt.plot(merged_data['GMT Time'], system_price, label='Actual')
plt.plot(pd.date_range(start=merged_data['GMT Time'].iloc[-1], periods=forecast_steps, freq='30T'), forecast, label='Forecast')
plt.title('System Price Forecast')
plt.xlabel('Time')
plt.ylabel('System Price (£/MWh)')
plt.legend()
plt.show()

# Calculate RMSE for ARIMA model
actual_values = system_price[-forecast_steps:]  # Actual values for the forecast period
rmse_arima = mean_squared_error(actual_values, forecast, squared=False)
print(f'RMSE for ARIMA Model: {rmse_arima:.4f}')

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
    ('xgb', XGBRegressor(n_estimators=100, random_state=42)),  # Using CPU version
    ('lgbm', LGBMRegressor(n_estimators=100, random_state=42))
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

# Combine ARIMA predictions with other model predictions
combined_predictions_price = (october_1_predictions_price + forecast) / 2

# Creating the output file
output = pd.DataFrame({
    'GMT_TIME': october_1_data['GMT Time'],
    'SYSTEM_PRICE': combined_predictions_price,
    'NIV_OUTTURN': october_1_predictions_niv
})

output.to_csv('predictions_october_1_2024.csv', index=False)
print(output.head())