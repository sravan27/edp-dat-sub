# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Energy Market Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    try:
        balancing = pd.read_csv('data/balancing_data.csv')
        demand = pd.read_csv('data/demand_load_data.csv')
        generation = pd.read_csv('data/generation_data.csv')
        price = pd.read_csv('data/price_data.csv')
        
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
        
        # Handle missing values
        merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')
        
        return merged_data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox(
    'Select Page',
    ['Executive Summary', 'Market Analysis', 'Technical Deep Dive', 'Business Calculator', 'Predictions']
)

# Load data
data = load_data()

if data is None:
    st.error("Failed to load data. Please check data files and try again.")
    st.stop()

# Executive Summary Page
if page == 'Executive Summary':
    st.title('Energy Market Intelligence Dashboard')
    st.subheader('Executive Summary')
    
    # Key Performance Indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_price = data['System Price (ESO Outturn) - GB (£/MWh)'].mean()
        st.metric("Average System Price", f"£{avg_price:.2f}")
    with col2:
        avg_demand = data['Actual Total Load - GB (MW)'].mean()
        st.metric("Average Demand", f"{avg_demand:.0f} MW")
    with col3:
        price_volatility = data['System Price (ESO Outturn) - GB (£/MWh)'].std()
        st.metric("Price Volatility", f"£{price_volatility:.2f}")
    
    # Summary Charts
    st.subheader('Market Overview')
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(data, x='datetime', y='System Price (ESO Outturn) - GB (£/MWh)',
                     title='System Price Trends')
        st.plotly_chart(fig)
    
    with col2:
        fig = px.box(data, x='day_of_week', y='System Price (ESO Outturn) - GB (£/MWh)',
                     title='Price Distribution by Day')
        st.plotly_chart(fig)

# Market Analysis Page
elif page == 'Market Analysis':
    st.title('Market Analysis')
    
    # Time period selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', data['datetime'].min())
    with col2:
        end_date = st.date_input('End Date', data['datetime'].max())
    
    filtered_data = data[(data['datetime'].dt.date >= start_date) & 
                        (data['datetime'].dt.date <= end_date)]
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(['Price Analysis', 'Demand Patterns', 'Market Correlations'])
    
    with tab1:
        st.subheader('Price Analysis')
        fig = px.line(filtered_data, x='datetime', y='System Price (ESO Outturn) - GB (£/MWh)',
                     title='Price Trends')
        st.plotly_chart(fig)
        
        # Price distribution
        fig = px.histogram(filtered_data, x='System Price (ESO Outturn) - GB (£/MWh)',
                          title='Price Distribution')
        st.plotly_chart(fig)
    
    with tab2:
        st.subheader('Demand Patterns')
        # Hourly demand patterns
        hourly_demand = filtered_data.groupby('hour')['Actual Total Load - GB (MW)'].mean()
        fig = px.line(hourly_demand, title='Average Hourly Demand')
        st.plotly_chart(fig)
    
    with tab3:
        st.subheader('Market Correlations')
        correlation_matrix = filtered_data[[
            'System Price (ESO Outturn) - GB (£/MWh)',
            'Actual Total Load - GB (MW)',
            'NIV Outturn (+ve long) - GB (MW)'
        ]].corr()
        fig = px.imshow(correlation_matrix, title='Correlation Matrix')
        st.plotly_chart(fig)

# Technical Deep Dive
elif page == 'Technical Deep Dive':
    st.title('Technical Analysis')
    
    # Model explanation
    st.subheader('Time Series Decomposition')
    
    # Decompose time series
    price_series = data.set_index('datetime')['System Price (ESO Outturn) - GB (£/MWh)']
    decomposition = seasonal_decompose(price_series, period=24)
    
    # Plot components
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    plt.tight_layout()
    st.pyplot(fig)

# Business Calculator
elif page == 'Business Calculator':
    st.title('Business Impact Calculator')
    
    col1, col2 = st.columns(2)
    
    with col1:
        volume = st.number_input('Energy Volume (MWh)', min_value=0.0)
        price = st.number_input('Expected System Price (£/MWh)', 
                              value=float(data['System Price (ESO Outturn) - GB (£/MWh)'].mean()))
        
    with col2:
        margin = st.slider('Profit Margin (%)', 0, 100, 15)
        risk_factor = st.slider('Risk Factor (%)', -50, 50, 0)
    
    if st.button('Calculate'):
        base_revenue = volume * price
        profit = base_revenue * (margin/100)
        risk_adjusted_profit = profit * (1 + risk_factor/100)
        
        st.subheader('Results')
        col1, col2, col3 = st.columns(3)
        col1.metric('Revenue', f'£{base_revenue:,.2f}')
        col2.metric('Base Profit', f'£{profit:,.2f}')
        col3.metric('Risk-Adjusted Profit', f'£{risk_adjusted_profit:,.2f}')
        
        # Historical comparison
        st.subheader('Historical Context')
        hist_prices = data['System Price (ESO Outturn) - GB (£/MWh)']
        fig = px.histogram(hist_prices, title='Historical Price Distribution')
        st.plotly_chart(fig)
        
        percentile = (hist_prices < price).mean() * 100
        st.info(f'Your selected price is at the {percentile:.1f}th percentile of historical prices')

# Predictions Page
else:
    st.title('Price Predictions')
    
    # Model parameters
    st.sidebar.subheader('Model Parameters')
    p = st.sidebar.slider('AR Order (p)', 0, 5, 1)
    d = st.sidebar.slider('Difference Order (d)', 0, 2, 1)
    q = st.sidebar.slider('MA Order (q)', 0, 5, 1)
    
    if st.button('Generate Forecast'):
        with st.spinner('Training model...'):
            # Prepare data
            price_series = data['System Price (ESO Outturn) - GB (£/MWh)']
            
            # Train ARIMA model
            model = ARIMA(price_series, order=(p,d,q))
            results = model.fit()
            
            # Generate forecast
            forecast_steps = 24
            forecast = results.get_forecast(steps=forecast_steps)
            forecast_index = pd.date_range(start=data['datetime'].iloc[-1] + timedelta(hours=1), periods=forecast_steps, freq='H')
            forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)
            
            # Plot results
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['datetime'], y=price_series, mode='lines', name='Historical'))
            fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series, mode='lines', name='Forecast'))
            fig.update_layout(title='Price Forecast', xaxis_title='Time', yaxis_title='System Price (£/MWh)')
            st.plotly_chart(fig)
            
            # Model metrics
            st.subheader('Model Performance')
            st.write(f'AIC: {results.aic:.2f}')
            st.write(f'BIC: {results.bic:.2f}')

# Footer
st.sidebar.markdown('---')
st.sidebar.info('Created by Your Team Name')