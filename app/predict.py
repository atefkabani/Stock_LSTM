import argparse
import pickle
import os
import warnings

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models import load_model


def parse_args():
    p = argparse.ArgumentParser(description="LSTM stock predictor")
    p.add_argument("--date",    required=True,           help="Target date, YYYY-MM-DD")
    p.add_argument("--csv",     default="A.csv",         help="Path to price CSV")
    p.add_argument("--model",   default="models/lstm_model.keras")
    p.add_argument("--artifacts", default="models/lstm_artifacts.pkl")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        target_date = pd.to_datetime(args.date)
    except Exception:
        raise SystemExit(f"Invalid date '{args.date}'. Use YYYY-MM-DD.")

    model = load_model(args.model)
    with open(args.artifacts, "rb") as f:
        art = pickle.load(f)
    scaler   = art["scaler"]
    lookback = art["config"]["lookback"]

    print(f"lookback: {lookback}  |  target date: {target_date.date()}")

    df = (pd.read_csv(args.csv, parse_dates=["Date"])
            .sort_values("Date")
            .reset_index(drop=True))
    last_known    = df["Date"].iloc[-1]
    series_scaled = scaler.transform(df["Close"].values.reshape(-1, 1))

    # Case A: backtest (date inside df)
    if target_date <= last_known:
        idx = df.index[df["Date"] == target_date]
        if len(idx) == 0:
            raise SystemExit(f"{target_date.date()} not found in df.")
        i = idx[0]
        if i < lookback:
            raise SystemExit(f"Not enough history before {target_date.date()} "
                             f"(need {lookback}, have {i}).")
        window = series_scaled[i - lookback : i]
        yhat_scaled = model.predict(window.reshape(1, lookback, 1), verbose=0)[0, 0]
        yhat   = scaler.inverse_transform([[yhat_scaled]])[0, 0]
        actual = df["Close"].iloc[i]
        print(f"Predicted: ${yhat:.2f}  |  Actual: ${actual:.2f}  |  "
              f"Error: {yhat - actual:+.2f} (backtest)")
        return

    # Case B: forecast (date past last row)
    window = series_scaled[-lookback:].copy()
    steps  = pd.date_range(last_known + pd.Timedelta(days=1), target_date, freq="B")
    yhat_scaled = None
    for _ in steps:
        yhat_scaled = model.predict(window.reshape(1, lookback, 1), verbose=0)[0, 0]
        window = np.vstack([window[1:], [[yhat_scaled]]])
    yhat = scaler.inverse_transform([[yhat_scaled]])[0, 0]
    print(f"Predicted for {target_date.date()}: ${yhat:.2f} "
          f"(forecast, {len(steps)} steps ahead)")


if __name__ == "__main__":
    main()