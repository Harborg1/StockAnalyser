import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_stock_heatmap(ticker_symbol, start_date, end_date):
    """
    Create a heatmap showing the daily stock performance for a given stock.
    
    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT').
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
    """
    # Fetch stock data
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    
    if stock_data.empty:
        print(f"No data available for {ticker_symbol} between {start_date} and {end_date}.")
        return
    
    # Calculate daily percentage change
    stock_data['Daily Change (%)'] = stock_data['Close'].pct_change() * 100
    
    # Create a pivot table for the heatmap (group by year and month)
    stock_data['Year'] = stock_data.index.year
    stock_data['Month'] = stock_data.index.month
    stock_data['Day'] = stock_data.index.day
    pivot_table = stock_data.pivot_table(values='Daily Change (%)', 
                                         index='Month', 
                                         columns='Day', 
                                         aggfunc='mean')
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='RdYlGn', annot=True, fmt=".1f", linewidths=0.5, center=0)
    plt.title(f'{ticker_symbol} Daily Performance Heatmap ({start_date} to {end_date})', fontsize=16)
    plt.xlabel('Day of Month', fontsize=12)
    plt.ylabel('Month', fontsize=12)
    plt.tight_layout()
    plt.show()

# Example usage
create_stock_heatmap('AAPL', '2024-01-01', '2024-03-31')
