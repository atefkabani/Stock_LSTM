import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

CONFIG = {
    "lookback": 5,            # Past days used as input sequence
    "hidden_units": 64,       # LSTM units per layer
    "num_layers": 1,          # Stacked LSTM layers
    "dropout": 0.2,           # Dropout rate
    "epochs": 10,            # Max training epochs
    "batch_size": 16,         # Mini-batch size
    "learning_rate": 0.001,   # Initial Adam learning rate
    "test_size": 0.2,         # Fraction for test set (used in train_test_split)
    "forecast_start_date": "1999-11-18",      # Start date for forecasting
    "forecast_days": 10,      # Number of future days to forecast
    "output_dir":   "./models", # Directory to save model and plots
    "csv_path": "./dataset/A.csv" # Path to your CSV file
}

def load_data(csv_path=None):
    df = None
    """Load stock data from a CSV file or fall back to built-in sample data."""
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        # check if df has the required columns
        required_columns = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if not required_columns.issubset(df.columns):
            print(f"  ⚠️ CSV file is missing required columns: {required_columns - set(df.columns)}")
            return None
        print(f"  ✓ Loaded {len(df)} rows from {csv_path}")
        df = df.sort_values("Date").reset_index(drop=True)
    else:
        print("  ⚠️ CSV file not found.")
        
   
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def create_sequences(data, lookback):
    """Convert a 1-D array into (X, Y) sliding-window sequences for LSTM."""
    X, Y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)


def preprocess(df, lookback, test_size):
    """
    Scale prices to [0,1], build sequences, and split with train_test_split.
    """
    prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices)

    X, Y = create_sequences(scaled, lookback)

    # ── train_test_split (shuffle=False to preserve time order) ───────────
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, shuffle=False
    )

    # Reshape for LSTM: (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

    split_idx = len(X_train)

    print(f"  ✓ Sequences created  →  Train: {len(X_train)}  |  Test: {len(X_test)}")
    return X_train, X_test, Y_train, Y_test, scaler, split_idx


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL DEFINITION  (tensorflow.keras)
# ══════════════════════════════════════════════════════════════════════════════

def build_model(lookback, cfg):
    """Build a stacked LSTM model using tensorflow.keras Sequential API."""

    model = Sequential()
    model.add(Input(shape=(lookback, 1)))

       
    model.add(
        LSTM(
            units=cfg["hidden_units"],
            name= "lstm_1",
        )
    )
    model.add(Dropout(cfg["dropout"], name=f"dropout_1"))

    model.add(Dense(32, activation="relu", name="dense_hidden"))
    model.add(Dense(1, name="output"))

    model.compile(
        optimizer=Adam(learning_rate=cfg["learning_rate"]),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(model, X_train, Y_train, X_test, Y_test, cfg):
    """Train with EarlyStopping and ReduceLROnPlateau callbacks."""

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1),
    ]

    print("\n  Training …")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    tl = history.history["loss"]
    vl = history.history["val_loss"]
    total = len(tl)

    print(f"\n  {'Epoch':>7}  {'Train Loss':>12}  {'Val Loss':>12}")
    print("  " + "─" * 35)
    for i in range(0, total, max(1, total // 10)):
        print(f"  {i+1:>7}  {tl[i]:>12.6f}  {vl[i]:>12.6f}")
    print(f"  {total:>7}  {tl[-1]:>12.6f}  {vl[-1]:>12.6f}")
    print("  " + "─" * 35)
    print(f"  ✓ Finished after {total} epochs  (best val_loss: {min(vl):.6f})\n")

    return history


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_train, Y_train, X_test, Y_test, scaler):
    """Generate predictions, inverse-transform, and compute metrics."""

    train_pred = scaler.inverse_transform(model.predict(X_train, verbose=0)).flatten()
    test_pred  = scaler.inverse_transform(model.predict(X_test, verbose=0)).flatten()
    train_actual = scaler.inverse_transform(Y_train.reshape(-1, 1)).flatten()
    test_actual  = scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(test_actual, test_pred))
    mae  = mean_absolute_error(test_actual, test_pred)
    r2   = r2_score(test_actual, test_pred)
    mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100

    print("  ┌──────────────────────────────────────┐")
    print("  │         TEST SET METRICS              │")
    print("  ├──────────────────────────────────────┤")
    print(f"  │  RMSE  : {rmse:>10.4f}                 │")
    print(f"  │  MAE   : {mae:>10.4f}                 │")
    print(f"  │  R²    : {r2:>10.4f}                 │")
    print(f"  │  MAPE  : {mape:>9.2f}%                 │")
    print("  └──────────────────────────────────────┘")

    return {
        "train_pred": train_pred, "test_pred": test_pred,
        "train_actual": train_actual, "test_actual": test_actual,
        "rmse": rmse, "mae": mae, "r2": r2, "mape": mape,
    }
    
def main():

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print()
    print("  ══════════════════════════════════════════════")
    print("   LSTM Stock Market Predictor")
    print("   tensorflow.keras  +  sklearn.train_test_split")
    print("  ══════════════════════════════════════════════")
    print()

    df = load_data(CONFIG["csv_path"])
    # if df is None:
        
    #     print("  ✗ Failed to load data. Exiting.")
    #     return
    print(f"  Date range : {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}")
    print(f"  Price range: ${df['Close'].min():.2f} – ${df['Close'].max():.2f}\n")



    X_train, X_test, Y_train, Y_test, scaler, split_idx = preprocess(
        df, CONFIG["lookback"], CONFIG["test_size"]
    )

    model = build_model(CONFIG["lookback"], CONFIG)
    model.summary(print_fn=lambda s: print(f"  {s}"))

    #################################################################################################
    history = train_model(model, X_train, Y_train, X_test, Y_test, CONFIG)
        
    ##################################################################################################
    results = evaluate(model, X_train, Y_train, X_test, Y_test, scaler)

    scaled_all = scaler.transform(df["Close"].values.reshape(-1, 1)).flatten()
    last_seq = scaled_all[-CONFIG["lookback"]:]

    model.save(os.path.join(CONFIG["output_dir"], "lstm_model.keras"))
    
    import pickle
 
    # Save
    with open(os.path.join(CONFIG["output_dir"], "scaler.pkl"), "wb") as f:    # "wb" = write binary
        pickle.dump(scaler, f)

    # Load
    with open(os.path.join(CONFIG["output_dir"], "scaler.pkl"), "rb") as f:    # "rb" = read binary
        scaler = pickle.load(f)
        
    artifacts = {
        "scaler":   scaler,
        "lookback": CONFIG["lookback"],
        "config":   CONFIG,
    }

    with open(os.path.join(CONFIG["output_dir"], "lstm_artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
        
        
        
if __name__ == "__main__":
    main()        


