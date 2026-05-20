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
prices = df[['Close']].astype(float).values

# ----------------------------
# 2. Normalize
# ----------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices)

WINDOW = 100
FUTURE = 10
MODEL_FILE = "xauusd_lstm_model.h5"

# ----------------------------
# 3. Train model if not already saved
# ----------------------------
if os.path.exists(MODEL_FILE):
    print("Loading existing model...")
    model = load_model(MODEL_FILE)
else:
    print("Training new model...")
    # Prepare first 100 candles for training
    X_train = scaled[:WINDOW].reshape(1, WINDOW, 1)
    y_train = scaled[WINDOW:WINDOW+FUTURE].reshape(1, FUTURE)
    
    # Build lightweight LSTM
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(WINDOW,1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(FUTURE)
    ])
    model.compile(optimizer="adam", loss="mse")
    
    model.fit(X_train, y_train, epochs=10, verbose=1)
    model.save(MODEL_FILE)
    print(f"Model trained and saved to {MODEL_FILE}")

# ----------------------------
# 4. Rolling predictions
# ----------------------------
success = 0
fail = 0
total_tests = 0

for start in range(0, len(scaled)-WINDOW-FUTURE, FUTURE):
    X_input = scaled[start:start+WINDOW].reshape(1, WINDOW, 1)
    y_real = scaled[start+WINDOW:start+WINDOW+FUTURE].reshape(FUTURE)
    
    y_pred_scaled = model.predict(X_input)
    y_pred = y_pred_scaled[0]
    
    # Directional comparison
    for i in range(FUTURE):
        real_change = y_real[i] - scaled[start+WINDOW-1][0]
        pred_change = y_pred[i] - scaled[start+WINDOW-1][0]
        if (real_change >=0 and pred_change >=0) or (real_change <0 and pred_change <0):
            success +=1
        else:
            fail +=1
        total_tests +=1

print(f"\nTotal predictions: {total_tests}")
print(f"Success: {success}")
print(f"Fail: {fail}")
print(f"Accuracy: {success/total_tests*100:.2f}%")
