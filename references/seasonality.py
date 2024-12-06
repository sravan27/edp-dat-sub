# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the datasets
balancing_data = pd.read_csv('/Users/sravansridhar/Desktop/Edp_datathon/data/balancing_data.csv')
demand_load_data = pd.read_csv('/Users/sravansridhar/Desktop/Edp_datathon/data/demand_load_data.csv')
generation_data = pd.read_csv('/Users/sravansridhar/Desktop/Edp_datathon/data/generation_data.csv')
price_data = pd.read_csv('/Users/sravansridhar/Desktop/Edp_datathon/data/price_data.csv')

# Convert GMT Time to datetime
for df in [balancing_data, demand_load_data, generation_data, price_data]:
    df['GMT Time'] = pd.to_datetime(df['GMT Time'])

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

# If the series is not stationary, apply differencing
system_price_diff = system_price.diff().dropna()

# Plot the differenced series
plt.figure(figsize=(15, 8))
plt.plot(merged_data['GMT Time'][1:], system_price_diff)
plt.title('Differenced System Price Over Time')
plt.xlabel('Time')
plt.ylabel('Differenced System Price (£/MWh)')
plt.show()

# Plot ACF and PACF to determine p and q
plot_acf(system_price_diff, lags=50)
plt.show()

plot_pacf(system_price_diff, lags=50)
plt.show()
