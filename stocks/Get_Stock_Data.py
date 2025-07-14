from datetime import datetime
from stocks.constants import stock_market_holidays
from auxillary.pre_market import get_pre_market_price_ticker
from stocks.Get_Crypto_Data  import crypto_reader
from stocks.base_reader import MarketReaderBase
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import auxillary.sentiment as Sentiment
import json
import calendar
class stock_reader(MarketReaderBase):
    def __init__(self, day_details_callback = None):
        super().__init__()
        self.cr = crypto_reader()
        self.day_details_callback = day_details_callback 

    def get_sentiment(self, stock:str, start_date:str, end_date:str) -> tuple[float, list | None] | None:
        s = Sentiment.get_news_sentiment(stock, start_date, end_date)
        return s
    
    
    def get_data_for_day(self, year:int, month:int, day:int, stock:str) ->  pd.DataFrame | str:
        start_date, end_date = self.get_start_and_end_date(year,month)
        data = self.download_data(start_date,end_date,stock)
        if not data.empty:
            date_str = f'{year}-{month:02d}-{day:02d}'
            date = pd.Timestamp(date_str)
            if date in data.index:
                row = data.loc[date]
                return row
            else:
                return f"No trading data available for {stock} on {date_str}."
        else:
            # Return the error message from download_data
            return "No data found"
    
    def get_close_price(self, s, e, stock):
        data = self.download_data(s, e, stock)
        l_close = []
        # Check if data is a DataFrame or error message
        if not data.empty:
            l_close = round(data['Close'], 2).values.flatten().tolist()
            return l_close
        
        else:
            # Return the error message or handle it (e.g., log or raise an error)
            return data  # This will return the error message directly
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

    # Function to generate non-weekend weeks with 5 elements each
    def get_non_weekend_weeks(self, year:int, month:int) -> list[list[int]]:
        # Get the first and last day of the month
        _, num_days_in_month = calendar.monthrange(year, month)
        all_days = pd.date_range(start=f"{year}-{month:02d}-01", end=f"{year}-{month:02d}-{num_days_in_month}")

        # Filter out weekends (Saturday = 5, Sunday = 6)
        non_weekend_days = [day.day for day in all_days if day.weekday() < 5]

        # Split non-weekend days into chunks of 5
        weeks = [non_weekend_days[i:i+5] for i in range(0, len(non_weekend_days), 5)]
        return weeks
    

    def get_json_data(self, file_name:str):
         with open(f'json_folder\\{file_name}', 'r') as file:
            data = json.load(file)

         return data
    
    def create_month_calendar_view(self, year:int, month:int, stock:str, download:bool=False):

        plt.close('all')

        current_date = datetime.now()
        current_month = current_date.month

        earnings_data = self.get_json_data("stock_earnings.json")


        earnings_dates = {
            pd.Timestamp(item["date"]) for item in earnings_data if item["stock"] == stock
        }

        cpi_data_all = self.get_json_data("cpi.json")

        fomc_data_all = self.get_json_data("fomc.json")
        
        cpi_data = {
            pd.Timestamp(item["date"]) for item in cpi_data_all
        }

        fomc_data  = {
            pd.Timestamp(item["date"]) for item in fomc_data_all
        }

        fig, ax = plt.subplots(figsize=(9, 6))  # Reduce height to save space

        # Reduce margins
        plt.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95)  # Adjust margins

        start_date, end_date = self.get_start_and_end_date(year, month)

        stock_data = self.download_data(start_date, end_date, stock)

        if stock_data is None: 
            return 0
        
        # Extract the trading days from the data
        trading_days = stock_data.index

        # Get the price differences and percentage changes
        price_differences = self.get_price_or_percentage_change(year, month, stock, return_percentage=False)
        if not isinstance(price_differences, list):
            return price_differences

        percentage_changes = self.get_price_or_percentage_change(year, month, stock, return_percentage=True)

        price_range = self.get_price_change_per_month(year, month, stock)

        monthly_percentage_change = self.get_monthly_percentage_change(year, month, stock)

        daily_change = self.get_price_range_per_day(year, month, stock)

        month_weeks = self.get_non_weekend_weeks(year,month)

        stock_market_holidays_list = stock_market_holidays(year)
        pre_market_price = get_pre_market_price_ticker(stock)
        
        # Track the index for price differences and percentages
        idx = 0
        rectangles_by_day = {}

        if stock=="BTC-USD":
            self.cr.create_calendar_view(year,month,stock,download)
            if download:
                return plt
            else:
                return

        # Iterate over the weeks and days in the month
        for week_idx, week in enumerate(month_weeks):
            for day_idx, day in enumerate(week):
                if month!=current_month:
                    day_rect = plt.Rectangle((day_idx, -week_idx), 1, -1, color="black",linewidth=1.5)
                else: 
                    day_rect = plt.Rectangle((day_idx, -week_idx), 1, -1, color="white")

                ax.add_patch(day_rect)

                current_date = pd.Timestamp(year=year, month=month, day=day)

                if current_date.normalize() == pd.Timestamp.today().normalize() and current_date not in trading_days:
                        # Check if current_date is in the trading days
                    if pre_market_price is not None: 
                        percentage_change = round((pre_market_price / self.get_last_trading_day_close(year, month, stock)) * 100 - 100, 2)
                        if percentage_change > 0:
                                color = 'green'
                        elif percentage_change< 0:
                                color = 'red'
                        else:
                            color = 'grey'
                        
                        day_rect.set_facecolor(color)
                        # Store the rectangle in the dictionary with the day as the key
                        rectangles_by_day[day] = day_rect
                        # Add text for the day and percentage change
                        ax.text(day_idx + 0.5, -week_idx - 0.3, str(day), ha='center', va='center', fontsize=9, weight='bold')
                        ax.text(day_idx + 0.5, -week_idx - 1, f'{pre_market_price}', ha='center', va='bottom', fontsize=9)
                        ax.text(day_idx + 0.5, -week_idx - 0.7, f'{percentage_change}%', ha='center', va='center', fontsize=9)
                    
                if current_date in trading_days:
                        if idx < len(price_differences):
                            price_diff = price_differences[idx]
                            percentage_change = percentage_changes[idx]
                            day_change = daily_change[idx]
                        else:
                            price_diff = 0  # Default value if no price difference available
                            percentage_change = 0.0
                            day_change = 0.0

                        # Determine color
                        if price_diff > 0:
                            color = 'green'
                        elif price_diff < 0:
                            color = 'red'
                        else:
                            color = 'grey'

                        day_rect.set_facecolor(color)
                        # Store the rectangle in the dictionary with the day as the key
                        rectangles_by_day[day] = day_rect
                        # Add text for the day and percentage change
                        ax.text(day_idx + 0.5, -week_idx - 0.3, str(day), ha='center', va='center', fontsize=9, weight='bold')
                        ax.text(day_idx + 0.5, -week_idx - 0.7, f'{percentage_change}%', ha='center', va='center', fontsize=9)
                        ax.text(day_idx + 0.5, -week_idx - 1, f'{day_change}', ha='center', va='bottom', fontsize=9)
                        # Increment the index only for valid trading days
                        idx += 1

                elif current_date in earnings_dates:
                    day_rect.set_facecolor("purple")
                    ax.text(day_idx + 0.5, -week_idx - 0.3, str(day), ha='center', va='center', fontsize=10, weight='bold')
                    ax.text(day_idx + 0.5, -week_idx - 0.5, str(earnings_dates[current_date] + " " + "earnings date"),
                            ha='center', va='center', fontsize=7, weight='bold', color='black')

                elif current_date in stock_market_holidays_list:
                    # Place a grey square for the holiday
                    day_rect.set_facecolor("grey")
                    ax.text(day_idx + 0.5, -week_idx - 0.3, str(day), ha='center', va='center', fontsize=10, weight='bold')
                    ax.text(day_idx + 0.5, -week_idx - 0.5, str(stock_market_holidays_list[current_date]),
                            ha='center', va='center', fontsize=7, weight='bold', color='black')
                    
                elif current_date in cpi_data:
                     # Place a blue square for CPI data
                    day_rect.set_facecolor("blue")
                    ax.text(day_idx + 0.5, -week_idx - 0.3, str(day), ha='center', va='center', fontsize=10, weight='bold')
                    ax.text(day_idx + 0.5, -week_idx - 0.5, "CPI data date",
                            ha='center', va='center', fontsize=7, weight='bold', color='black')
                elif current_date in fomc_data:
                    # Place a yellow square for FOMC data
                    day_rect.set_facecolor("yellow")
                    ax.text(day_idx + 0.5, -week_idx - 0.3, str(day), ha='center', va='center', fontsize=10, weight='bold')
                    ax.text(day_idx + 0.5, -week_idx - 0.5, "FOMC meeting today or tomorrow",
                            ha='center', va='center', fontsize=6, weight='bold', color='black')
                    
        if month==current_month:
            # Add grid lines
            for i in range(1, 7):  # Vertical grid lines
                ax.axvline(i - 1, color='black', linewidth=1.5)
            for j in range(len(month_weeks)+1):  # Horizontal grid lines
                ax.axhline(-j, color='black', linewidth=1.5)

        # Set limits and labels
        ax.set_xlim(0, 5)
        ax.set_ylim(-len(month_weeks), 0)
        ax.axis('off')  # Turn off the axes

        # Set limits and labels
        ax.set_xlim(0, 5)
        ax.set_ylim(-len(month_weeks), 0)
        ax.axis('off')  # Turn off the axes
        ax.set_title(
            f'{stock} Performance in {calendar.month_name[month]} {year}\n'
            f'Price Range: ${price_range[0]} - ${price_range[1]}\n'
            f'Overall Percentage Change: {float(monthly_percentage_change.iloc[0])}%',
            fontsize=12
        )
        # Function to be called when clicking on a day
        def on_click(event):
            if event.inaxes == ax:
                for day, patch in rectangles_by_day.items():
                    if patch.contains_point((event.x, event.y)):
                        day_clicked = day
                        # Call the callback function if available
                        if self.day_details_callback:
                            self.day_details_callback(year, month, day_clicked, stock)
                        break  # Stop after the first match

        # Add the event listener to the figure
        fig.canvas.mpl_connect('button_press_event', on_click)
        if download:  # If we are downloading the file
            return plt
        else:  # Show the plot if we want to get a monthly view
            plt.show()


    