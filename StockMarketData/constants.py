
import pandas as pd
class DateConstants:
    def stock_market_holidays(self,year):
        if year == 2025:
            stock_market_holidays = {
                pd.Timestamp(year=2025, month=1, day=1): "New Year's Day",
                pd.Timestamp(year=2025, month=1, day=20): "Martin Luther King Jr. Day",
                pd.Timestamp(year=2025, month=2, day=17): "Presidents' Day",
                pd.Timestamp(year=2025, month=4, day=18): "Good Friday",
                pd.Timestamp(year=2025, month=5, day=26): "Memorial Day",
                pd.Timestamp(year=2025, month=6, day=19): "Juneteenth",
                pd.Timestamp(year=2025, month=7, day=4): "Independence Day",
                pd.Timestamp(year=2025, month=9, day=1): "Labor Day",
                pd.Timestamp(year=2025, month=11, day=11): "Veterans Day",
                pd.Timestamp(year=2025, month=11, day=27): "Thanksgiving Day",
                pd.Timestamp(year=2025, month=12, day=25): "Christmas Day",
            }
        elif year == 2024:
                    stock_market_holidays = {
                    pd.Timestamp(year=year, month=1, day=1): "New Year's Day",
                    pd.Timestamp(year=year, month=1, day=15): "Martin Luther King Jr. Day",
                    pd.Timestamp(year=year, month=2, day=19): "Presidents' Day",
                    pd.Timestamp(year=year, month=3, day=29): "Good friday",
                    pd.Timestamp(year=year, month=5, day=27): "Memorial Day",
                    pd.Timestamp(year=year, month = 6, day = 19): "JuneTeenth",
                    pd.Timestamp(year=year, month=7, day=4): "Independence Day",
                    pd.Timestamp(year=year, month=9, day=2): "Labor Day",
                    pd.Timestamp(year=year, month=11, day=11): "Veterans Day",
                    pd.Timestamp(year=year, month=11, day=28): "Thanksgiving Day",
                    pd.Timestamp(year=year, month=12, day=25): "Christmas Day", }
                    
        elif year == 2023:
            stock_market_holidays = {
            pd.Timestamp(year=2023, month=1, day=1): "New Year's Day",
            pd.Timestamp(year=2023, month=1, day=16): "Martin Luther King Jr. Day",
            pd.Timestamp(year=2023, month=2, day=20): "Presidents' Day",
            pd.Timestamp(year=2023, month=5, day=29): "Memorial Day",
            pd.Timestamp(year=2023, month=6, day=19): "Juneteenth",
            pd.Timestamp(year=2023, month=7, day=4): "Independence Day",
            pd.Timestamp(year=2023, month=9, day=4): "Labor Day",
            pd.Timestamp(year=2023, month=11, day=10): "Veterans Day (observed)",
            pd.Timestamp(year=2023, month=11, day=23): "Thanksgiving Day",
            pd.Timestamp(year=2023, month=12, day=25): "Christmas Day",
        }
           
        elif year == 2022:
            stock_market_holidays = {
            pd.Timestamp(year=2022, month=1, day=1): "New Year's Day",
            pd.Timestamp(year=2022, month=1, day=17): "Martin Luther King Jr. Day",
            pd.Timestamp(year=2022, month=2, day=21): "Presidents' Day",
            pd.Timestamp(year=2022, month=5, day=30): "Memorial Day",
            pd.Timestamp(year=2022, month=6, day=20): "Juneteenth (observed)",
            pd.Timestamp(year=2022, month=7, day=4): "Independence Day",
            pd.Timestamp(year=2022, month=9, day=5): "Labor Day",
            pd.Timestamp(year=2022, month=11, day=11): "Veterans Day",
            pd.Timestamp(year=2022, month=11, day=24): "Thanksgiving Day",
            pd.Timestamp(year=2022, month=12, day=25): "Christmas Day (observed on Dec 26)",
        }
        return stock_market_holidays
    

