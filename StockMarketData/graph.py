# import yfinance as yf

# def get_hourly_stock_data(ticker_symbol, start_date, end_date):
#     """
#     Fetch stock data with hourly interval for a specific date range.

#     Args:
#         ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT').
#         start_date (str): The start date in 'YYYY-MM-DD' format.
#         end_date (str): The end date in 'YYYY-MM-DD' format.

#     Returns:
#         pandas.DataFrame: A DataFrame containing stock data with hourly intervals.
#     """
#     # Download stock data with hourly interval
#     stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1h')
    
#     if stock_data.empty:
#         print(f"No data available for {ticker_symbol} between {start_date} and {end_date}.")
#     else:
#         print(stock_data.head())  # Print the first few rows of the data
    
#     return stock_data

# # Example usage
# stock_data = get_hourly_stock_data('AAPL', '2024-01-03', '2024-01-04')
