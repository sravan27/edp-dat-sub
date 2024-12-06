# pages/preprocessing.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def show():
    st.title('Data Preprocessing & Feature Engineering')
    
    try:
        # Load raw data
        balancing = pd.read_csv('data/balancing_data.csv')
        demand = pd.read_csv('data/demand_load_data.csv')
        generation = pd.read_csv('data/generation_data.csv')
        price = pd.read_csv('data/price_data.csv')
        
        # Display raw data samples
        with st.expander("View Raw Data Samples"):
            st.write("Balancing Data:", balancing.head())
            st.write("Demand Data:", demand.head())
            st.write("Generation Data:", generation.head())
            st.write("Price Data:", price.head())
        
        if st.button("Process & Engineer Features"):
            with st.spinner("Processing data..."):
                # Merge datasets
                merged_data = pd.merge(balancing, demand, on='GMT Time')
                merged_data = pd.merge(merged_data, generation, on='GMT Time')
                merged_data = pd.merge(merged_data, price, on='GMT Time')
                
                # Feature Engineering
                merged_data['datetime'] = pd.to_datetime(merged_data['GMT Time'])
                merged_data['hour'] = merged_data['datetime'].dt.hour
                merged_data['day_of_week'] = merged_data['datetime'].dt.dayofweek
                merged_data['month'] = merged_data['datetime'].dt.month
                merged_data['is_weekend'] = merged_data['day_of_week'].isin([5,6]).astype(int)
                
                # Save processed data
                merged_data.to_csv('data/merged_data_processed.csv', index=False)
                st.success("Data processed successfully!")
                
                # Display feature info
                st.subheader("Engineered Features Preview")
                st.write(merged_data[['datetime', 'hour', 'day_of_week', 'month', 'is_weekend']].head())
                
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")