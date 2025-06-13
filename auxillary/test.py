import yfinance as yf

stock = yf.download("CLSK", period="5d", interval="1d", auto_adjust=True, progress=False)

# Compute percent change over entire Close series
stock['day_return_pct'] = stock['Close'].pct_change() * 100

latest_day_return = round(stock['day_return_pct'].iloc[-1].sum(),2)

print(latest_day_return)

stock['cum_return_3d'] = stock['day_return_pct'].rolling(window=3).sum()

latest_cum_return_3d = round(stock['cum_return_3d'].iloc[-1],2)


print(latest_cum_return_3d)