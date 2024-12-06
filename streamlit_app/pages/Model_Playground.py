# pages/model_playground.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import plotly.express as px

def show():
    st.title('Model Playground')
    
    data = pd.read_csv('data/merged_data_processed.csv')
    
    # Model Configuration
    st.sidebar.subheader("Model Settings")
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["Random Forest", "XGBoost"]
    )
    
    target = st.sidebar.selectbox(
        "Select Target Variable",
        ["system_price_eso_outturn_gb_mwh", "niv_outturn_ve_long_gb_mw"]
    )
    
    # Feature Selection
    features = st.multiselect(
        "Select Features",
        data.columns.drop([target, 'datetime', 'GMT Time']),
        default=data.columns[:5].drop([target, 'datetime', 'GMT Time'] if target in data.columns[:5] else None)
    )
    
    if st.button("Train Model"):
        X = data[features]
        y = data[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100)
        else:
            model = xgb.XGBRegressor()
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Results
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        col1.metric("RMSE", f"{rmse:.2f}")
        col2.metric("R² Score", f"{r2:.2f}")
        
        # Feature Importance Plot
        importances = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(importances, x='feature', y='importance',
                    title='Feature Importance')
        st.plotly_chart(fig)