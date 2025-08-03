import yfinance as yf

df = yf.download('TSLA', start='2024-09-17', end='2025-10-07', interval='1h')
df.index = df.index.tz_convert('UTC')  # Make sure the index is timezone-aware

print(df.loc['2025-03-21 16:30:00+00:00']["Close"])
print(df.loc['2024-10-01 15:30:00+00:00']["Close"])

#2024-09-18 16:30:00+00:00,2024-10-01 15:30:00+00:00,563.719970703125,568.77001953125,5.050048828125,1
