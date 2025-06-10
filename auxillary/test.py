import yfinance as yf

stock = yf.download("CLSK")

# Compute percent change over entire Close series
stock['day_return_pct'] = stock['Close'].pct_change() * 100

latest_day_return = stock['day_return_pct'].iloc[-1].sum()

print(latest_day_return)

stock['cum_return_3d'] = stock['day_return_pct'].rolling(window=3).sum()

print(stock['cum_return_3d'].iloc[-1])
