
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from stocks.base_reader import MarketReaderBase
import calendar
class crypto_reader(MarketReaderBase):
    def __init__(self):
        super().__init__()

    def get_full_weeks(self, year, month):
        _, days = calendar.monthrange(year, month)
        all_days = [i for i in range(1, days + 1)]
        weeks = [all_days[i:i+7] for i in range(0, len(all_days), 7)]
        if weeks and len(weeks[-1]) < 7:
            weeks[-1] += [None] * (7 - len(weeks[-1]))
        return weeks
    
    def create_calendar_view(self, year, month, stock,download):
        plt.close('all')
        price_range = self.get_price_change_per_month(year,month,stock)
      
        s, e = self.get_start_and_end_date(year, month)
        data = self.download_data(s, e, stock)
        if data.empty:
            print("No data.")
            return

        prices = self.get_price_range_per_day(year, month, stock)
        weeks = self.get_full_weeks(year, month)

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95)

        idx = 0
        for w_idx, week in enumerate(weeks):
            for d_idx, day in enumerate(week):
                if day is None:
                    continue

                current_date = pd.Timestamp(year=year, month=month, day=day)
                if current_date in data.index:
                    open_price, close_price = prices[idx]
                    color = (
                        'green' if close_price > open_price else
                        'red' if close_price < open_price else 'grey'
                    )
                    rect = plt.Rectangle((d_idx, -w_idx), 1, -1, color=color)
                    ax.add_patch(rect)
                    ax.text(d_idx + 0.5, -w_idx - 0.3, str(day), ha='center', va='center', fontsize=9)
                    ax.text(d_idx + 0.5, -w_idx - 0.7, f'({open_price}, {close_price})',
                            ha='center', va='center', fontsize=7.3)
                idx += 1
        for i in range(1, 8):
            ax.axvline(i - 1, color='black', linewidth=1)
        for j in range(len(weeks) + 1):
            ax.axhline(-j, color='black', linewidth=1)

        ax.set_xlim(0, 7)
        ax.set_ylim(-len(weeks), 0)
        ax.get_yaxis().set_visible(False)
        ax.set_title(
            f'{stock} Performance in {calendar.month_name[month]} {year}\n'
            f'Price Range: ${price_range[0]} - ${price_range[1]}\n'
            f'20 day moving average: {self.get_moving_average(self.start_date,self.end_date,stock,True)}\n'
            f'50 day moving average: {self.get_moving_average(self.start_date,self.end_date,stock,False)}',
            fontsize=10
        )
        if download:
            return plt
        else:
            plt.show()
