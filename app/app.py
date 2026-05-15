import os, pickle, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="LSTM Stock Predictor", page_icon="📈", layout="centered")
st.title("📈 LSTM Stock Predictor")

# ---------- Load once, cache ----------
@st.cache_resource
def load_artifacts(model_path="lstm_model.keras", pkl_path="lstm_artifacts.pkl"):
    model = load_model(model_path)
    with open(pkl_path, "rb") as f:
        art = pickle.load(f)
    return model, art["scaler"], art["config"]["lookback"]

@st.cache_data
def load_prices(csv_path="A.csv"):
    return (pd.read_csv(csv_path, parse_dates=["Date"])
              .sort_values("Date").reset_index(drop=True))

model, scaler, lookback = load_artifacts()
df = load_prices()

# ---------- Sidebar inputs ----------
st.sidebar.header("Inputs")
csv_path     = st.sidebar.text_input("CSV path", "A.csv")
target_date  = st.sidebar.date_input(
    "Target date",
    value=df["Date"].iloc[-1].date() + pd.Timedelta(days=1),
    min_value=df["Date"].iloc[lookback].date(),
)
go = st.sidebar.button("Predict", type="primary")

st.caption(f"Data: {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}  "
           f"|  lookback = {lookback}  |  rows = {len(df)}")

# ---------- Prediction ----------
def predict(target_date):
    target = pd.to_datetime(target_date)
    last_known = df["Date"].iloc[-1]
    series_scaled = scaler.transform(df["Close"].values.reshape(-1, 1))

    if target <= last_known:
        idx = df.index[df["Date"] == target]
        if len(idx) == 0:
            return {"error": f"{target.date()} not in dataset"}
        i = idx[0]
        if i < lookback:
            return {"error": f"Need {lookback} rows of history; have {i}"}
        window = series_scaled[i - lookback : i]
        yhat_scaled = model.predict(window.reshape(1, lookback, 1), verbose=0)[0, 0]
        yhat = float(scaler.inverse_transform([[yhat_scaled]])[0, 0])
        return {"mode": "backtest", "predicted": yhat,
                "actual": float(df["Close"].iloc[i]), "date": target}

    window = series_scaled[-lookback:].copy()
    steps = pd.date_range(last_known + pd.Timedelta(days=1), target, freq="B")
    yhat_scaled = None
    for _ in steps:
        yhat_scaled = model.predict(window.reshape(1, lookback, 1), verbose=0)[0, 0]
        window = np.vstack([window[1:], [[yhat_scaled]]])
    yhat = float(scaler.inverse_transform([[yhat_scaled]])[0, 0])
    return {"mode": "forecast", "predicted": yhat,
            "steps": len(steps), "date": target}

if go:
    with st.spinner("Predicting..."):
        result = predict(target_date)

    if "error" in result:
        st.error(result["error"])
    elif result["mode"] == "backtest":
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted", f"${result['predicted']:.2f}")
        c2.metric("Actual",    f"${result['actual']:.2f}")
        c3.metric("Error",     f"${result['predicted'] - result['actual']:+.2f}")
        st.success(f"Backtest for {result['date'].date()}")
    else:
        st.metric("Predicted Close", f"${result['predicted']:.2f}")
        st.info(f"Forecast for {result['date'].date()} "
                f"({result['steps']} business days ahead)")

    # Chart: history + the predicted point
    chart_df = df[["Date", "Close"]].rename(columns={"Close": "Actual"}).set_index("Date")
    chart_df.loc[result["date"], "Predicted"] = result["predicted"]
    st.line_chart(chart_df.tail(180))