# ==================================================
# XAUUSD LSTM Prediction (Mobile Optimized)
# File: XAU_1Month_data.csv
# Input: Previous 100 candles
# Output: Next 10 predicted candles
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings("ignore")

# ================================
# 1. LOAD CSV
# ================================

file_path = "XAU_1Month_data.csv"  # <-- Your CSV file

try:
    df = pd.read_csv(file_path, sep=";")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Convert date column
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Use only Close for prediction
prices = df[['Close']].astype(float)
print("Candles loaded:", len(prices))

# ================================
# 2. NORMALIZE DATA
# ================================

scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices)

WINDOW = 100  # previous candles
FUTURE = 10   # predict next 10 candles

X, y = [], []
for i in range(len(scaled) - WINDOW - FUTURE):
    X.append(scaled[i:i+WINDOW, 0])
    y.append(scaled[i+WINDOW:i+WINDOW+FUTURE, 0])

X = np.array(X)
y = np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))  # LSTM input

print("Dataset ready. X shape:", X.shape, "Y shape:", y.shape)

# ================================
# 3. TRAIN / TEST SPLIT
# ================================

split = int(len(X) * 0.85)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ================================
# 4. BUILD LIGHTWEIGHT LSTM MODEL (Mobile-Friendly)
# ================================

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW,1)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(FUTURE)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# ================================
# 5. TRAIN MODEL (fast for mobile)
# ================================

print("Training started ...")
history = model.fit(
    X_train, y_train,
    epochs=5,           # small for mobile test, increase later
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
print("Training finished!")

# ================================
# 6. PREDICT NEXT 10 CANDLES
# ================================

last_sequence = scaled[-WINDOW:].reshape(1, WINDOW, 1)
pred_scaled = model.predict(last_sequence)
prediction = scaler.inverse_transform(pred_scaled)[0]

print("\n======= NEXT 10 CANDLES PREDICTION =======")
for i,p in enumerate(prediction,1):
    print(f"Candle +{i}: {p:.2f} USD")
print("==========================================\n")

# ================================
# 7. PLOT RESULTS
# ================================

plt.figure(figsize=(12,6))
plt.plot(prices[-400:].values, label="Actual Price")
future_index = range(len(prices), len(prices)+FUTURE)
plt.plot(future_index, prediction, label="Predicted Future")
plt.title("XAUUSD Next 10 Candles Prediction")
plt.xlabel("Candles")
plt.ylabel("Price")
plt.legend()
plt.show()
