import streamlit as st
import numpy as np
import pandas as pd
from binance.client import Client
from scipy.stats import skew, kurtosis
import plotly.graph_objs as go
import time

client = Client()

def get_price_data(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1MINUTE, limit=200):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    closes = [float(k[4]) for k in klines]
    return pd.DataFrame(closes, columns=['close'])

def get_signal(df, window=100):
    log_returns = np.log(df / df.shift(1)).dropna()
    past = log_returns[-2*window:-window]
    future = log_returns[-window:]

    σ_now, σ_future = np.std(past), np.std(future)
    S_now, C_now = skew(past), kurtosis(past, fisher=False)
    S_future, C_future = skew(future), kurtosis(future, fisher=False)

    def jb(S, C, n): return (n/6) * (S**2 + ((C - 3)**2)/4)
    JB_now, JB_future = jb(S_now, C_now, window), jb(S_future, C_future, window)

    vol_signal = np.sign(σ_future - σ_now)
    jb_signal = np.sign(JB_future - JB_now)

    if vol_signal > 0 and jb_signal > 0:
        return "LONG (BUY)", σ_now, σ_future, JB_now, JB_future
    elif vol_signal < 0 and jb_signal < 0:
        return "SHORT (SELL)", σ_now, σ_future, JB_now, JB_future
    else:
        return "WAIT / NO SIGNAL", σ_now, σ_future, JB_now, JB_future

# Dashboard
st.set_page_config(page_title="BTCUSDT Signal Bot", layout="wide")
st.title("📊 BTCUSDT.P Real-Time Trading Signal Bot")
st.markdown("Built using Volatility + Skewness + Kurtosis")

placeholder = st.empty()

while True:
    df = get_price_data()
    signal, σ_now, σ_future, JB_now, JB_future = get_signal(df['close'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['close'], mode='lines', name="Price"))
    fig.update_layout(title="BTCUSDT Price (last 200 min)", xaxis_title="Minutes", yaxis_title="Price")

    with placeholder.container():
        st.subheader(f"🚦 Signal: {signal}")
        st.metric("Volatility (Now)", f"{σ_now:.5f}")
        st.metric("Volatility (Future)", f"{σ_future:.5f}")
        st.metric("JB Now", f"{JB_now:.2f}")
        st.metric("JB Future", f"{JB_future:.2f}")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Auto-refreshes every 60s.")

    time.sleep(60)
