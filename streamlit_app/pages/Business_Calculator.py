# business_calculator.py

import streamlit as st

def show():
    st.title("Business Calculator")
    st.markdown("## Calculate Business Metrics")

    # User inputs
    st.markdown("### Input Parameters")
    energy_sold = st.number_input("Energy Sold (MWh)", min_value=0.0, value=1000.0)
    system_price = st.number_input("System Price (£/MWh)", min_value=0.0, value=50.0)

    # Calculate revenue
    revenue = energy_sold * system_price

    st.markdown("### Results")
    st.write(f"Total Revenue: £{revenue:,.2f}")