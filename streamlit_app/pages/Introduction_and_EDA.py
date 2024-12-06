# pages/introduction_and_eda.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

def show():
    st.title('Introduction & Exploratory Data Analysis (EDA)')

    # Load data
    @st.cache_data
    def load_data():
        try:
            data = pd.read_csv('data/merged_data_processed.csv')
            # Ensure 'datetime' column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(data['datetime']):
                data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
            return data
        except FileNotFoundError:
            st.error("Processed data not found. Please run the Data Preprocessing page first.")
            return None

    data = load_data()

    if data is not None:
        # Display basic statistics
        st.subheader("Data Overview")
        st.write("**Available Columns:**", data.columns.tolist())
        st.write(data.head())

        # Display Metrics
        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_price = data['system_price'].mean()
            st.metric("Average System Price", f"£{avg_price:.2f}")
        with col2:
            max_price = data['system_price'].max()
            st.metric("Max System Price", f"£{max_price:.2f}")
        with col3:
            avg_niv = data['niv_outturn'].mean()
            st.metric("Average NIV", f"{avg_niv:.2f} MW")

        # Distribution Plots
        st.subheader("Data Distributions")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distribution of System Price**")
            fig1, ax1 = plt.subplots()
            sns.histplot(data['system_price'].dropna(), bins=50, kde=True, ax=ax1)
            ax1.set_xlabel('System Price (£/MWh)')
            st.pyplot(fig1)
        
        with col2:
            st.markdown("**Distribution of Net Imbalance Volume (NIV)**")
            fig2, ax2 = plt.subplots()
            sns.histplot(data['niv_outturn'].dropna(), bins=50, kde=True, ax=ax2)
            ax2.set_xlabel('NIV (MW)')
            st.pyplot(fig2)

        # Boxplots by Hour
        st.subheader("System Price and NIV by Hour of the Day")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**System Price by Hour**")
            fig3, ax3 = plt.subplots()
            sns.boxplot(x='hour', y='system_price', data=data, ax=ax3)
            ax3.set_xlabel('Hour of Day')
            ax3.set_ylabel('System Price (£/MWh)')
            st.pyplot(fig3)

        with col2:
            st.markdown("**NIV by Hour**")
            fig4, ax4 = plt.subplots()
            sns.boxplot(x='hour', y='niv_outturn', data=data, ax=ax4)
            ax4.set_xlabel('Hour of Day')
            ax4.set_ylabel('NIV (MW)')
            st.pyplot(fig4)

        # Correlation Matrix
        st.subheader("Correlation Analysis")
        fig5, ax5 = plt.subplots(figsize=(12, 10))
        correlation_matrix = data.select_dtypes(include=['float', 'int']).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax5)
        st.pyplot(fig5)

        # Seasonality Analysis
        st.subheader("Seasonality Analysis")

        # Augmented Dickey-Fuller Test
        def adf_test(series, title=''):
            st.markdown(f"**{title}**")
            result = adfuller(series.dropna(), autolag='AIC')
            st.write(f'ADF Statistic: {result[0]:.4f}')
            st.write(f'p-value: {result[1]:.4f}')
            for key, value in result[4].items():
                st.write(f'Critical Value ({key}): {value:.4f}')
            if result[1] <= 0.05:
                st.success("Reject the null hypothesis - The data is stationary.")
            else:
                st.warning("Fail to reject the null hypothesis - The data is non-stationary.")

        st.markdown("**Augmented Dickey-Fuller Test**")
        adf_test(data['system_price'], "System Price (ESO Outturn) - GB (£/MWh)")
        adf_test(data['niv_outturn'], "NIV Outturn (+ve long) - GB (MW)")

        # ACF and PACF Plots
        st.markdown("**Autocorrelation and Partial Autocorrelation**")

        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        fig6, ax6 = plt.subplots(2, 1, figsize=(15, 10))
        plot_acf(data['system_price'].dropna(), ax=ax6[0], lags=50)
        plot_pacf(data['system_price'].dropna(), ax=ax6[1], lags=50)
        ax6[0].set_title('ACF - System Price')
        ax6[1].set_title('PACF - System Price')
        st.pyplot(fig6)

        fig7, ax7 = plt.subplots(2, 1, figsize=(15, 10))
        plot_acf(data['niv_outturn'].dropna(), ax=ax7[0], lags=50)
        plot_pacf(data['niv_outturn'].dropna(), ax=ax7[1], lags=50)
        ax7[0].set_title('ACF - NIV Outturn')
        ax7[1].set_title('PACF - NIV Outturn')
        st.pyplot(fig7)

        # Seasonality Decomposition
        st.markdown("**Seasonal Decomposition**")

        from statsmodels.tsa.seasonal import seasonal_decompose

        def decompose_series(series, title, period):
            decomposition = seasonal_decompose(series.dropna(), model='additive', period=period)
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            decomposition.observed.plot(ax=axes[0], legend=False)
            axes[0].set_ylabel('Observed')
            decomposition.trend.plot(ax=axes[1], legend=False)
            axes[1].set_ylabel('Trend')
            decomposition.seasonal.plot(ax=axes[2], legend=False)
            axes[2].set_ylabel('Seasonal')
            decomposition.resid.plot(ax=axes[3], legend=False)
            axes[3].set_ylabel('Residual')
            fig.suptitle(title)
            st.pyplot(fig)

        # Determine period based on data frequency
        # Assuming hourly data with daily seasonality
        period = 24  # Adjust if frequency is different

        st.markdown("**Seasonal Decomposition - System Price**")
        decompose_series(data['system_price'], 'System Price Decomposition', period)

        st.markdown("**Seasonal Decomposition - NIV Outturn**")
        decompose_series(data['niv_outturn'], 'NIV Outturn Decomposition', period)