# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

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

# Plotting the distribution of System Price
plt.figure(figsize=(10, 6))
sns.histplot(merged_data['System Price (ESO Outturn) - GB (£/MWh)'], bins=50, kde=True)
plt.title('Distribution of System Price')
plt.xlabel('System Price (£/MWh)')
plt.ylabel('Frequency')
plt.show()

# Plotting the distribution of Net Imbalance Volume
plt.figure(figsize=(10, 6))
sns.histplot(merged_data['NIV Outturn (+ve long) - GB (MW)'], bins=50, kde=True)
plt.title('Distribution of Net Imbalance Volume')
plt.xlabel('Net Imbalance Volume (MW)')
plt.ylabel('Frequency')
plt.show()

# Plotting the correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = merged_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Time series decomposition for System Price
result = seasonal_decompose(merged_data.set_index('GMT Time')['System Price (ESO Outturn) - GB (£/MWh)'], model='additive', period=48)
result.plot()
plt.show()

# Time series decomposition for Net Imbalance Volume
result = seasonal_decompose(merged_data.set_index('GMT Time')['NIV Outturn (+ve long) - GB (MW)'], model='additive', period=48)
result.plot()
plt.show()

# Plotting System Price over time
plt.figure(figsize=(15, 8))
plt.plot(merged_data['GMT Time'], merged_data['System Price (ESO Outturn) - GB (£/MWh)'])
plt.title('System Price Over Time')
plt.xlabel('Time')
plt.ylabel('System Price (£/MWh)')
plt.show()

# Plotting Net Imbalance Volume over time
plt.figure(figsize=(15, 8))
plt.plot(merged_data['GMT Time'], merged_data['NIV Outturn (+ve long) - GB (MW)'])
plt.title('Net Imbalance Volume Over Time')
plt.xlabel('Time')
plt.ylabel('Net Imbalance Volume (MW)')
plt.show()

# Plotting average System Price by hour of the day
plt.figure(figsize=(10, 6))
sns.boxplot(x='Hour', y='System Price (ESO Outturn) - GB (£/MWh)', data=merged_data)
plt.title('Average System Price by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('System Price (£/MWh)')
plt.show()

# Plotting average Net Imbalance Volume by hour of the day
plt.figure(figsize=(10, 6))
sns.boxplot(x='Hour', y='NIV Outturn (+ve long) - GB (MW)', data=merged_data)
plt.title('Average Net Imbalance Volume by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Net Imbalance Volume (MW)')
plt.show()

# Plotting average System Price by day of the week
plt.figure(figsize=(10, 6))
sns.boxplot(x='DayOfWeek', y='System Price (ESO Outturn) - GB (£/MWh)', data=merged_data)
plt.title('Average System Price by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('System Price (£/MWh)')
plt.show()

# Plotting average Net Imbalance Volume by day of the week
plt.figure(figsize=(10, 6))
sns.boxplot(x='DayOfWeek', y='NIV Outturn (+ve long) - GB (MW)', data=merged_data)
plt.title('Average Net Imbalance Volume by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Net Imbalance Volume (MW)')
plt.show()

# Plotting average System Price by month
plt.figure(figsize=(10, 6))
sns.boxplot(x='Month', y='System Price (ESO Outturn) - GB (£/MWh)', data=merged_data)
plt.title('Average System Price by Month')
plt.xlabel('Month')
plt.ylabel('System Price (£/MWh)')
plt.show()

# Plotting average Net Imbalance Volume by month
plt.figure(figsize=(10, 6))
sns.boxplot(x='Month', y='NIV Outturn (+ve long) - GB (MW)', data=merged_data)
plt.title('Average Net Imbalance Volume by Month')
plt.xlabel('Month')
plt.ylabel('Net Imbalance Volume (MW)')
plt.show()