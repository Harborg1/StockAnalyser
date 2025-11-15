
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

# Download data
df = yf.download("SPY", start="2025-01-01", end="2025-11-14")
prices = df["Close"].dropna()

# Ensure datetime index and remove missing days, but DO NOT reindex with frequency
prices.index = pd.to_datetime(prices.index)
predicted = []
actual = []

window = 10  # must be >= 7 for AR(3)

for i in range(window, len(prices)):
    # rolling window of real data, no artificial business days
    train_data = prices.iloc[i-window:i]

    # sanity check: skip windows containing NaNs
    if train_data.isna().any().item():
        continue

    model = AutoReg(train_data, lags=3, old_names=False)
    results = model.fit()

    # predict next day
    pred = results.predict(start=len(train_data), end=len(train_data))

    predicted.append(pred.values[0])
    actual.append(prices.iloc[i])


# convert to arrays
predicted = np.array(predicted)
actual = np.array(actual)

# error metrics
mae = np.mean(np.abs(predicted - actual))
mape = np.mean(np.abs((predicted - actual) / actual)) * 100

print(f"MAE:  {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
