import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_simulation import generate_demand
from src.forecast_lstm import train_for_sku
from src.optimizer_or_tools import optimize_orders
from src.langchain_report import generate_report

st.set_page_config(page_title='Supply Chain Optimizer', layout='wide')
st.title('Supply Chain Optimizer — Demo')

DATA = Path(__file__).resolve().parents[1] / 'data'
if not (DATA / 'demo_demand.csv').exists():
    st.info('Generating demo data...')
    generate_demand()

uploaded = st.file_uploader('Upload demand CSV (optional)', type=['csv'])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv(DATA / 'demo_demand.csv')

sku = st.selectbox('Choose SKU', df['sku'].unique())
series = df[df
