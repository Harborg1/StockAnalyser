import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
ticker = "SPY"
start_date = "2015-01-01"
end_date = "2025-09-06"
initial_capital = 10000.0

# Download adjusted close prices
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
close = data['Close']

# Compute equity curve for buy-and-hold strategy
returns = close.pct_change().fillna(0)
equity_curve = (1 + returns).cumprod() * initial_capital

# Plot
plt.figure(figsize=(10, 5))
plt.plot(equity_curve.index, equity_curve.values)
plt.title(f"{ticker} Buy & Hold Equity Curve\n{start_date} to {end_date}")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.tight_layout()
plt.show()
