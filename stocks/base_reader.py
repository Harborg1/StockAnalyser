import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
# A superclass for the stock- and crypto class respectively
class MarketReaderBase: 
    def __init__(self):
        self.cache = {}
        self.date_cache =  {}
        self.monthly_date = {}
        self.start_date = "2024-01-01"
        self.end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    def get_start_and_end_date(self, year, month):
        key = (year, month)
        if key in self.monthly_date:
            return self.monthly_date[key]
        start_date = f'{year}-{month:02d}-01'
        if month == 12:
            end_date = f'{year + 1}-01-01'
        else:
            end_date = f'{year}-{month + 1:02d}-01'
        self.monthly_date[key] = (start_date, end_date)
        return start_date, end_date
    
    def get_moving_average(self, s:int , e:int, stock:str, ma20:bool):
        # Download the data using the existing method
        data = self.download_data(s, e, stock)
        if data.empty:
            return f"No data found for {stock} between {s} and {e}."
        # Choose the window based on the ma20 flag
        window = 20 if ma20 else 50

        # Compute the rolling mean (moving average) of the 'Close' column
        moving_avg_series = data['Close'].rolling(window=window).mean().round(2).iloc[-1].item()
        # Return the moving average values as a list
        return moving_avg_series


    def is_valid_date_range(self, s:int) -> bool:
        today = pd.Timestamp.now()
        return pd.Timestamp(s) <= today  # Return True or False only
    

    def download_data(self, s:str, e:str, stock:str) -> pd.DataFrame | tuple[str,int,int]:
        # Generate a unique key for caching
        cache_key = (stock, s, e)
        date_cache_key = s

        if date_cache_key in self.date_cache:
            return pd.DataFrame()
        
        # Check if the data is already in the cache
        if cache_key in self.cache:
            #print(f"Using cached data for {stock} between {s} and {e}.")
            return self.cache[cache_key]
        
        if not self.is_valid_date_range(s):
            self.date_cache[s] = True
            print("Invalid date range")
            return pd.DataFrame()
        
        try:
            # Attempt to download with a timeout
            spy_ohlc_df = yf.download(stock, start=s, end=e,auto_adjust=True,progress=False)
            if not spy_ohlc_df.empty:
                #print("Downloading data...")
                self.cache[cache_key] = spy_ohlc_df
                #print(spy_ohlc_df)
                return spy_ohlc_df
            else:
                print(f"No data found for {stock} between {s} and {e}.")
                return pd.DataFrame()
        except Exception as ex:
            print(f"Download error for {stock}: {ex}")
            return pd.DataFrame()

    def get_price_range_per_day(self, year,month,stock):
        start_date, end_date = self.get_start_and_end_date(year, month)
        data = self.download_data(start_date,end_date,stock)
        l = []
        if stock=="BTC-USD":
            for i in range(len(data)):
              l.append((int(data['Open'].iloc[i].item()),int(data['Close'].iloc[i].item())))
        else:
            for i in range(len(data)):
                l.append((round(data['Open'].iloc[i].item(),2),round(data['Close'].iloc[i].item(),2)))

        return l
    
    def get_price_change_per_month(self,year:int,month:int,stock:str) -> tuple[float, float]:
        start_date, end_date = self.get_start_and_end_date(year,month)
        data = self.download_data(start_date,end_date,stock)
        min_val = 10**9
        max_val = 0
        for i in range(len(data)):
            min_val = min(min_val, data['Open'].iloc[i].item(), data['Close'].iloc[i].item())

        for i in range(len(data)):
            max_val = max(max_val, data['Open'].iloc[i].item(),data['Close'].iloc[i].item())
        if stock=="BTC-USD":
            return int(min_val),int(max_val)
        return round(min_val,2),round(max_val,2)
    

    def get_price_or_percentage_change(self, year: int, month: int, stock: str, return_percentage: bool = False) -> list[float]:
        start_date, end_date = self.get_start_and_end_date(year, month)
        l_close = self.get_close_price(start_date, end_date, stock)
        if not isinstance(l_close, list):
            return l_close  # Return error message if get_close_price fails
        result = []
        previous_month_end = pd.Timestamp(start_date) - pd.DateOffset(days=1)
        last_month_close = self.get_last_trading_day_close(previous_month_end.year, previous_month_end.month, stock)

        for i, current_close in enumerate(l_close):
            if i == 0:
                prev_close = last_month_close
            else:
                prev_close = l_close[i - 1]

            if prev_close:
                delta = current_close - prev_close
                change_value = round((delta / prev_close) * 100, 2) if return_percentage else round(delta, 2)
            else:
                change_value = 0.0

            result.append(change_value)

        return result

    def get_monthly_percentage_change(self, year:int, month:int, stock:str) -> float:
        start_date, end_date = self.get_start_and_end_date(year,month)

        # Download the data within the date range
        data = self.download_data(start_date, end_date,stock)
        
        # Ensure there are valid trading days
        if len(data) > 1:
            first_open = data['Open'][stock].iloc[0]
            last_close = data['Close'][stock].iloc[-1]

            # Calculate the percentage change from the first to the last trading day
            percentage_change = round(((last_close / first_open) - 1) * 100, 2)
        
        #Grab the first index of the list since we need to get the last month's closing price to calculate the percentage change.
        elif len(data==1):
            percentage_change = self.get_price_or_percentage_change(year,month,stock,return_percentage=True)[0]

        else:

            percentage_change = 0.0  # No trading days in the month

        return percentage_change
    
    def get_last_trading_day_close(self, year:int, month:int, stock:str) -> pd.DataFrame | tuple[str, int, int]:
        start_date, end_date = self.get_start_and_end_date(year, month)
        # Download the data within the date range
        data = self.download_data(start_date, end_date, stock)
        return data['Close'].iloc[-1].item()
    
 

   