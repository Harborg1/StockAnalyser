
from Get_Stock_Data import stock_reader
from datetime import datetime
import tkinter as tk
from tkinter import ttk            
import tkinter as tk
import pandas as pd
import yfinance as yf
import webbrowser
import os
import json
from datetime import datetime
from web_scraper import web_scraper
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
class App:
    def __init__(self, root):
        self.stock_reader_instance = stock_reader(day_details_callback=self.open_day_details_window)
        self.path = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
        self.root = root
        self.root.title("Stock Data Calendar Viewer")
        # Main frame for the layout
        self.main_frame = tk.Frame(root, padx=10, pady=10)
        self.main_frame.grid(row=0, column=0)
        self.populate_main_screen()
        self.get_sentiment_data()
    def populate_main_screen(self):
        """Populates the main screen with the original content."""
        # Clear the main frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        # Stock selection
        stock_frame = tk.Frame(self.main_frame, pady=5)
        stock_frame.grid(row=0, column=0, sticky="w")
        tk.Label(stock_frame, text="Select stock:").grid(row=0, column=0, padx=5, pady=5)
        self.stock_entry = ttk.Combobox(stock_frame, values=["NVO", "TSLA", "CLSK", "NVDA","PLTR","SPY"], state="readonly")
        self.stock_entry.grid(row=0, column=1, padx=5)
        self.stock_entry.set("CLSK")
        self.web_scraper_instance = web_scraper(self.stock_entry.get())

        self.navigate_button = tk.Button(
        stock_frame, text="→", font=("Arial", 12), command=self.re_populate_screen)
        self.navigate_button.grid(row=0, column=5, padx=2, pady=2)
        # Year and Month selection
        date_frame = tk.Frame(self.main_frame, pady=5)
        date_frame.grid(row=1, column=0, sticky="w")
        tk.Label(date_frame, text="Select Year:").grid(row=0, column=0, padx=5, pady=5)
        self.year_entry = ttk.Combobox(date_frame, values=[2021,2022, 2023, 2024, 2025], state="readonly")
        self.year_entry.grid(row=0, column=1, padx=5)
        self.year_entry.set(datetime.now().year)
        tk.Label(date_frame, text="Select Month:").grid(row=0, column=2, padx=5, pady=5)
        self.month_entry = ttk.Combobox(date_frame, values=list(range(1, 13)), state="readonly")
        self.month_entry.grid(row=0, column=3, padx=5)
        self.month_entry.set(datetime.now().month)
        # Buttons for calendar actions
        button_frame = tk.Frame(self.main_frame, pady=10,padx=10)
        button_frame.grid(row=2, column=0, sticky="w")
        button_frame.grid_columnconfigure(1, minsize=120)
        self.show_button = tk.Button(button_frame, text="Show Calendar", command=self.show_calendar)
        self.show_button.grid(row=1, column=0, padx=5, pady=5)
        self.sentiment_text = tk.Text(button_frame, height=1, width=20)
        self.sentiment_text.grid(row=1, column=2, padx=5, pady=5)
        action_frame = tk.Frame(self.main_frame, pady=10)
        action_frame.grid(row=3, column=0, sticky="w")
        self.download_button = tk.Button(action_frame, text="Download", command=self.download_calendar)
        self.download_button.grid(row=0, column=0, padx=5, pady=5)
        self.fetch_news_button = tk.Button(action_frame, text="Fetch news", command=lambda: self.fetch_news())
        self.fetch_news_button.grid(row=0, column=2, padx=5, pady=5)
    def re_populate_screen(self) -> list:
        """Replaces the main screen with the second screen, showing portfolio details."""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        stocks = ["TSLA", "NVDA", "CLSK","DKIGI.CO","PLTR"]
        usd_dkk = int(self.stock_reader_instance.get_last_trading_day_close(datetime.now().year, datetime.now().month,"DKK=X"))

        # Portfolio data
        portfolio = {
            stocks[0]: {"shares": 45, "price": int(self.stock_reader_instance.get_last_trading_day_close(datetime.now().year,datetime.now().month, stocks[0]))},
            stocks[1]: {"shares": 91, "price": int(self.stock_reader_instance.get_last_trading_day_close(datetime.now().year, datetime.now().month, stocks[1]))},  
            stocks[2]: {"shares": 600, "price": int(self.stock_reader_instance.get_last_trading_day_close(datetime.now().year, datetime.now().month, stocks[2]))},
            stocks[3]: {"shares": 993, "price": int(self.stock_reader_instance.get_last_trading_day_close(datetime.now().year, datetime.now().month, stocks[3]))/usd_dkk},
            stocks[4]: {"shares": 105, "price": int(self.stock_reader_instance.get_last_trading_day_close(datetime.now().year, datetime.now().month, stocks[4]))}
        }
        # Calculate portfolio value
        total_value = sum(stock["shares"] * stock["price"] for stock in portfolio.values())
        # Back button (Top-left corner)
        tk.Button(
            self.main_frame, text="←", font=("Arial", 12), command=self.populate_main_screen
        ).grid(row=0, column=0, sticky="w", padx=10, pady=10)  # Align to top-left corner
        # Display portfolio value
        tk.Label(
            self.main_frame, 
            text=f"Portfolio Value: ${total_value:,.2f}", 
            font=("Arial", 16), 
            fg="green"
        ).grid(row=3, column=0)
        
        # Prepare data for the pie chart
        labels = portfolio.keys()
        sizes = [stock["shares"] * stock["price"] for stock in portfolio.values()]
        colors = ["#FF5733", "#33FF57", "#3357FF","#FF33A8","Brown"]  # Assign unique colors for each stock

        # Create the pie chart
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors
        )
        ax.axis("equal")  # Equal aspect ratio ensures the pie chart is circular.
        # Embed the chart in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, self.main_frame)
        canvas.get_tk_widget().grid(row=6, column=0, pady=10)
        # Close the figure after rendering to prevent multiple figures from being displayed
        plt.close(fig)

    def get_news_links_for_month(self, year, month) -> list:
        try:
            # Load the news data from the JSON file
            with open('json_folder\\article_links.json', 'r') as f:
                news_data = json.load(f)
            # List to store links for the specified month and year
            month_links = []
            # Iterate through each article to filter by year and month
            for article in news_data:
                # Parse the date from the JSON into a datetime object
                article_date = datetime.strptime(article['date'], "%B %d, %Y")
                # Check if the year and month match the parameters
                if article_date.year == year and article_date.month == month:
                    month_links.append(article['link'])  # Add the URL to the list

            # print("Returning month links...")
            # print(month_links)
            return month_links
        
        except FileNotFoundError:
            print("The news_releases.json file was not found.")
            return []
        except json.JSONDecodeError:
            print("Error decoding the JSON file.")
            return []
        
    def open_day_details_window(self, year, month, day, stock):
        if os.path.exists(self.path):
            webbrowser.register("edge", None, webbrowser.BackgroundBrowser(self.path))
        else:
            print("Microsoft Edge not found at default paths.")
        """Opens a new window to show detailed stock data for the selected day."""
        btc_mined = self.web_scraper_instance.calculate_total_btc(self.web_scraper_instance.bitcoin_data_2024)
        day_window = tk.Toplevel(self.root)
        day_window.title(f"Stock Details for {stock} - {year}-{month:02d}-{day:02d}")
        data = self.stock_reader_instance.get_data_for_day(year, month, day, stock)
        start_date=f"{year}-{month}-{day}"
        sentiment_data = self.stock_reader_instance.get_sentiment(stock, start_date,start_date)
        ma20 = self.stock_reader_instance.get_moving_average(self.stock_reader_instance.start_date, self.stock_reader_instance.end_date, stock, ma20=True)
        ma50 = self.stock_reader_instance.get_moving_average(self.stock_reader_instance.start_date, self.stock_reader_instance.end_date, stock, ma20=False)
        #print("Sentiment data:", sentiment_data)
        if stock =="CLSK":

            links = self.get_news_links_for_month(year,month)
        
        else:
            links = []


        if isinstance(data, pd.Series):
            # Extract the required data and convert them to standard Python types
            open_price = float(data['Open'].iloc[0])
            high_price = float(data['High'].iloc[0])
            low_price = float(data['Low'].iloc[0])
            close_price = float(data['Close'].iloc[0])
            volume = int(data['Volume'].iloc[0])
            # Display the data with proper formatting
            tk.Label(day_window, text=f"Date: {year}-{month:02d}-{day:02d}", font=("Arial", 14)).pack(pady=5)
            tk.Label(day_window, text=f"Stock: {stock}", font=("Arial", 14)).pack(pady=5)
            tk.Label(day_window, text=f"Open Price: ${open_price:,.2f}", font=("Arial", 12)).pack(pady=2)
            tk.Label(day_window, text=f"High Price: ${high_price:,.2f}", font=("Arial", 12)).pack(pady=2)
            tk.Label(day_window, text=f"Low Price: ${low_price:,.2f}", font=("Arial", 12)).pack(pady=2)
            tk.Label(day_window, text=f"Close Price: ${close_price:,.2f}", font=("Arial", 12)).pack(pady=2)
            tk.Label(day_window, text=f"Volume: {volume:,}", font=("Arial", 12)).pack(pady=2)
            tk.Label(day_window, text=f"20-day ma: {ma20:,}", font=("Arial", 12)).pack(pady=2)
            tk.Label(day_window, text=f"50-day ma: {ma50:,}", font=("Arial", 12)).pack(pady=2)
            if stock=="CLSK":
             tk.Label(day_window, text=f"BTC mined: {btc_mined:,}", font=("Arial", 12)).pack(pady=2)
             link = tk.Label(day_window, text="Click me to see the total network hashrate", font=("Arial", 10), fg="blue", cursor="hand2")
             link.pack(pady=1)
             link.bind("<Button-1>", lambda e, url="https://minerstat.com/coin/BTC/network-hashrate": webbrowser.get("edge").open(url))
            
            # Handle sentiment data
            if sentiment_data !=None:
                sentiment, urls = sentiment_data
                tk.Label(day_window, text=f"Sentiment Score: {sentiment:.2f}", font=("Arial", 12)).pack(pady=2)
                if urls:
                    tk.Label(day_window, text="Articles:", font=("Arial", 12)).pack(pady=2)
                    # Display the article URLs
                    for url in urls:
                        link = tk.Label(day_window, text=url, font=("Arial", 10), fg="blue", cursor="hand2")
                        link.pack(pady=1)
                        link.bind("<Button-1>", lambda e, url=url: webbrowser.get("edge").open(url))

                for lnk in links:
                    link = tk.Label(day_window, text=lnk, font=("Arial", 10), fg="blue", cursor="hand2")
                    link.pack(pady=1)
                    link.bind("<Button-1>", lambda e, url=lnk: webbrowser.get("edge").open(url))
                
    def fetch_news(self):
         # Fetch the selected stock dynamically from the dropdown
        stock = str(self.stock_entry.get())
        #self.web_scraper_instance = web_scraper(stock)
        # Only scrape articles for CLSK
        if stock == "CLSK":
            self.web_scraper_instance.scrape_articles()
            self.web_scraper_instance.scrape_bitcoin_address()
        # Scrape earnings for any stock
        self.web_scraper_instance.scrape_earnings()


    def get_sentiment_data(self):
        try:
            # Check if the data is in the .json file...
            with open('json_folder\\feargreed.json', 'r') as f:
                sentiment_data = json.load(f)
                if sentiment_data["date"]==datetime.now().strftime("%Y-%m-%d"):
                    self.sentiment_text.insert(tk.END, f'Sentiment value: {sentiment_data["fear_greed_index"]}')
        except:
            #Call the webscraper to get sentiment data if the data is not present...
            print("Could not retrieve the data")

        if sentiment_data["date"]!=datetime.now().strftime("%Y-%m-%d"):
            sentiment_value = self.web_scraper_instance.scrape_fear_greed_index(self.web_scraper_instance.sentiment_url)
            if sentiment_value:
                    self.sentiment_text.insert(tk.END, f'Sentiment value: {sentiment_value}')
            else:
                    self.sentiment_text.insert(tk.END, "Could not retrieve sentiment data.")

    def show_calendar(self): 
        year = int(self.year_entry.get())
        month = int(self.month_entry.get())
        stock = str(self.stock_entry.get())
        # Draw the calendar
        self.stock_reader_instance.create_month_calendar_view(year, month, stock, download=False)

    def download_calendar(self):
        year = int(self.year_entry.get())
        month = int(self.month_entry.get())
        stock = str(self.stock_entry.get())
        year_str = str(year) 
        month_str = str(month)
        # Download the calendar
        self.figure = self.stock_reader_instance.create_month_calendar_view(year, month, stock, download=True)  # Get the figure
        txt = stock+"_"+year_str+"_"+month_str+".pdf"
        self.figure.savefig(txt)  # Save the figure

# Initialize Tkinter
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    # app.get_news_links_for_month(2024,10)
    root.mainloop()
