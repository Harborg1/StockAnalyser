import yfinance as yf
import numpy as np 
import matplotlib.pyplot as plt


# Fetch data
stock = yf.download('TSLA')
stock.columns = stock.columns.get_level_values(0)


# Add day index
stock['day'] = np.arange(1, len(stock) + 1)
stock = stock[['day', 'Open', 'High', 'Low', 'Close']]

# Moving averages with shift to avoid lookahead bias
stock['9-day'] = stock['Close'].rolling(9).mean().shift()
stock['21-day'] = stock['Close'].rolling(21).mean().shift()

# Generate trading signals
stock['signal'] = np.where(stock['9-day'] > stock['21-day'], 1, 0)
stock['signal'] = np.where(stock['9-day'] < stock['21-day'], -1, stock['signal'])

# Drop NaNs
stock.dropna(inplace=True)

# Calculate log returns and strategy performance
stock['return'] = np.log(stock['Close']).diff()
stock['system_return'] = stock['signal'] * stock['return']
stock['entry'] = stock['signal'].diff()

# Filter data for 2025
stock_2025 = stock[stock.index >= '2025-01-01']

# Plot 2025 signals and prices
plt.figure(figsize=(12, 6))
plt.grid(True, alpha=.3)
plt.plot(stock_2025['Close'], label='TSLA')
plt.plot(stock_2025['9-day'], label='9-day')
plt.plot(stock_2025['21-day'], label='21-day')
plt.plot(stock_2025.loc[stock_2025.entry == 2].index, stock_2025['9-day'][stock_2025.entry == 2], '^',
         color='g', markersize=12, label='Buy')
plt.plot(stock_2025.loc[stock_2025.entry == -2].index, stock_2025['21-day'][stock_2025.entry == -2], 'v',
         color='r', markersize=12, label='Sell')
plt.legend(loc=2)
plt.title("CLSK Price and Trading Signals - 2025")
plt.show()

# Plot cumulative returns for 2025
plt.figure(figsize=(12, 6))
plt.plot(np.exp(stock_2025['return'].cumsum()), label='Buy/Hold')
plt.plot(np.exp(stock_2025['system_return'].cumsum()), label='System')
plt.legend(loc=2)
plt.grid(True, alpha=.3)
plt.title("Cumulative Returns in 2025")
plt.show()

# Print final returns for 2025
buy_hold_return = np.exp(stock_2025['return'].sum()) - 1
system_return = np.exp(stock_2025['system_return'].sum()) - 1
print("2025 Buy & Hold return:", buy_hold_return)
print("2025 System return:", system_return)
