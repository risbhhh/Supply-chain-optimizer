import streamlit as st
import pandas as pd
from pathlib import Path

from src.data_simulation import generate_demand_data
from src.utils import load_demand_data, plot_demand
from src.forecast_lstm import train_lstm_forecaster
from src.optimizer_or_tools import optimize_inventory
from src.langchain_report import generate_report

st.set_page_config(page_title="Supply Chain Optimizer", layout="wide")
st.title("📦 Supply Chain Optimizer Dashboard")

DATA_PATH = Path("data/demo_demand.csv")

# --- Sidebar controls ---
st.sidebar.header("Settings")
generate_new = st.sidebar.button("🔄 Generate synthetic data")
horizon = st.sidebar.slider("Forecast horizon (days)", 5, 30, 7)

# --- Data loading / generation ---
if generate_new or not DATA_PATH.exists():
    df = generate_demand_data()
    df.to_csv(DATA_PATH, index=False)
    st.sidebar.success("Generated fresh synthetic data ✅")
else:
    df = load_demand_data(DATA_PATH)

# --- Show demand data ---
st.subheader("📊 Demand Data")
st.dataframe(df.tail(10))

with st.expander("Show demand plot"):
    plot_demand(df)

# --- Forecasting ---
st.subheader("🔮 Forecasting with LSTM")
series = df["demand"].values
forecast = train_lstm_forecaster(series, epochs=30, forecast_horizon=horizon)
st.write(f"**Next {horizon} days forecast:**", forecast.tolist())

# --- Optimization ---
st.subheader("⚙️ Inventory Optimization")
on_hand = st.number_input("Current inventory (units)", min_value=0, value=100, step=10)

if st.button("Run Optimization"):
    results = optimize_inventory(
        forecasts=forecast.tolist(),
        on_hand=on_hand,
    )
    st.success("Optimization complete ✅")
    st.json(results)

    # --- LLM Report ---
    with st.spinner("Generating executive summary..."):
        try:
            report = generate_report(results)
            st.subheader("📝 Executive Report")
            st.write(report)
        except Exception as e:
            st.error(f"LLM report failed: {e}")
