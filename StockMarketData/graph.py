import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

def get_date_x_days_before(date_string, num_days_before):
    date_object = dt.datetime.strptime(date_string, "%Y-%m-%d")
    new_date = date_object - dt.timedelta(days=num_days_before)
    new_date_string = new_date.strftime("%Y-%m-%d")
    return new_date_string

stock = "CLSK"
start_date = "2024-10-01"
end_date = "2024-12-31"
num_periods = 20
# Get a few days before the start date to accommodate the period size
start_date_x_days_before = get_date_x_days_before(start_date, num_periods * 2)

# Grab the stock data
stock_data = yf.download(stock, start=start_date_x_days_before, end=end_date)

# Compute the simple moving average (SMA)
stock_data["SMA"] = stock_data["Close"].rolling(window=num_periods).mean()

# Now that we calculated the SMA, we can remove the dates before the actual start date that we want.
stock_data = stock_data[start_date:]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data["Close"], label='Close Price', color='blue', linewidth=1.5)
plt.plot(stock_data.index, stock_data["SMA"], label='20-Day SMA', color='orange', linewidth=1.5)

# Adding titles and labels
plt.title(f'{stock} Closing Prices and {num_periods}-Day SMA (2020)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid()

# Show the plot
plt.show()

# Print the SMA values
print(stock_data["SMA"])

