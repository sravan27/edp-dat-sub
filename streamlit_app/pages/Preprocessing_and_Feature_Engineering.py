# preprocessing_and_feature_engineering.py

import streamlit as st
import pandas as pd

def show():
    st.title("Preprocessing and Feature Engineering")
    st.markdown("## Data Preprocessing Steps")

    # Load raw datasets
    balancing_data = pd.read_csv('data/balancing_data.csv', parse_dates=['GMT Time'])
    demand_load_data = pd.read_csv('data/demand_load_data.csv', parse_dates=['GMT Time'])
    generation_data = pd.read_csv('data/generation_data.csv', parse_dates=['GMT Time'])
    price_data = pd.read_csv('data/price_data.csv', parse_dates=['GMT Time'])

    # Merge datasets
    merged_data = balancing_data.merge(demand_load_data, on='GMT Time') \
                                .merge(generation_data, on='GMT Time') \
                                .merge(price_data, on='GMT Time')

    # Handle missing values
    merged_data.fillna(method='ffill', inplace=True)
    merged_data.fillna(method='bfill', inplace=True)

    # Feature engineering
    merged_data['hour'] = merged_data['GMT Time'].dt.hour
    merged_data['day_of_week'] = merged_data['GMT Time'].dt.dayofweek
    merged_data['month'] = merged_data['GMT Time'].dt.month
    merged_data['is_weekend'] = (merged_data['day_of_week'] >= 5).astype(int)

    # Rename columns for consistency
    merged_data.rename(columns={
        'GMT Time': 'datetime',
        # Add any other necessary renames here
    }, inplace=True)

    # Save preprocessed data
    merged_data.to_csv('data/merged_data_processed.csv', index=False)

    st.markdown("### Preprocessed Data Sample")
    st.write(merged_data.head())