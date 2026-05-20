import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# ----------------------------
# 1. Load CSV
# ----------------------------
file_path = "XAU_1Month_data.csv"
df = pd.read_csv(file_path, sep=";")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Use OHLC + Volume as features
features = ['Open', 'High', 'Low', 'Close', 'Volume']
df[features] = df[features].astype(float)
data = df[features].values

# ----------------------------
# 2. Normalize all features
# ----------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# ----------------------------
# 3. Parameters
# ----------------------------
WINDOW = 100      # past candles as input
FUTURE = 10       # predict next 10 Close prices
MODEL_FILE = "xauusd_lstm_ohlcv_model.keras"

# ----------------------------
# 4. Set TRAIN_START
# ----------------------------
TRAIN_START = 50  # change to any starting index

# Safety check
if TRAIN_START + WINDOW + FUTURE > len(scaled):
    raise ValueError(f"TRAIN_START too large! Maximum allowed: {len(scaled) - WINDOW - FUTURE}")

# ----------------------------
# 5. Prepare training data
# ----------------------------
X_train = scaled[TRAIN_START:TRAIN_START+WINDOW]  # shape: (WINDOW, 5 features)
X_train = X_train.reshape(1, WINDOW, len(features))
y_train = scaled[TRAIN_START+WINDOW:TRAIN_START+WINDOW+FUTURE, 3]  # predict only Close
y_train = y_train.reshape(1, FUTURE)

# ----------------------------
# 6. Train or load model
# ----------------------------
if os.path.exists(MODEL_FILE):
    print("Loading existing model...")
    model = load_model(MODEL_FILE)
else:
    print(f"Training new model on candles {TRAIN_START}-{TRAIN_START+WINDOW-1}...")
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(WINDOW, len(features))),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(FUTURE)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=20, verbose=1)
    model.save(MODEL_FILE)
    print(f"Model trained and saved to {MODEL_FILE}")

# ----------------------------
# 7. Rolling predictions (directional)
# ----------------------------
success = 0
fail = 0
total_tests = 0

for start in range(TRAIN_START, len(scaled)-WINDOW-FUTURE, FUTURE):
    X_input = scaled[start:start+WINDOW].reshape(1, WINDOW, len(features))
    y_real = scaled[start+WINDOW:start+WINDOW+FUTURE, 3].reshape(FUTURE)  # actual Close
    
    y_pred_scaled = model.predict(X_input)
    y_pred = y_pred_scaled[0]
    
    for i in range(FUTURE):
        real_change = y_real[i] - scaled[start+WINDOW-1, 3]
        pred_change = y_pred[i] - scaled[start+WINDOW-1, 3]
        if (real_change >=0 and pred_change >=0) or (real_change <0 and pred_change <0):
            success +=1
        else:
            fail +=1
        total_tests +=1

# ----------------------------
# 8. Print results
# ----------------------------
print(f"\nTotal predictions: {total_tests}")
print(f"Success: {success}")
print(f"Fail: {fail}")
print(f"Accuracy: {success/total_tests*100:.2f}%")

