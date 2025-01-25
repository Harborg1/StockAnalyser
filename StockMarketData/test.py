import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import yfinance as yf  # For downloading stock data

# Step 1: Load Stock Data
ticker = "AAPL"  # Example: Apple stock
data = yf.download(ticker, start="2024-01-01", end="2025-01-01")
data['Returns'] = data['Close'].pct_change()  # Add daily returns as a feature
data.dropna(inplace=True)  # Drop rows with NaN values

# Step 2: Prepare Features and Target
X = data[['Open', 'High', 'Low', 'Volume', 'Returns']]  # Features
y = data['Close']  # Target variable (closing price)

# Step 3: Split Data (chronological split)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 4: Train SVR Model
model = SVR(kernel='rbf', C=1.0, epsilon=0.1)  # RBF kernel
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Step 7: Visualize Predictions (Optional)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label="Actual Prices")
plt.plot(y_test.index, y_pred, label="Predicted Prices")
plt.title(f"{ticker} Stock Price Prediction Using SVR")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
