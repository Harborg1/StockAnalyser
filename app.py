from typing import Dict, List, Optional, Tuple, Any
from stocks.Get_Stock_Data import stock_reader
from stocks.Get_Crypto_Data import crypto_reader
from datetime import datetime
import tkinter as tk
from tkinter import ttk            
import tkinter as tk
import pandas as pd
import webbrowser
from pre_market import get_pre_market_price_ticker
import os
import json
from datetime import datetime
from webscrapers.web_scraper import web_scraper
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class App:
    """Main application class for the Stock Data Calendar Viewer.
    This class handles the GUI and business logic for viewing and analyzing stock data.
    It provides functionality for viewing stock calendars, portfolio details, and news data.
    """
    def __init__(self, root: tk.Tk) -> None:
        """Initialize the application.
        Args:
            root: The main Tkinter window instance.
        """
        self.stock_reader_instance = stock_reader(day_details_callback=self.open_day_details_window)
        self.crypto_reader_instance = crypto_reader()
        self.path = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
        self.root = root
        self.root.title("Stock Data Calendar Viewer")
        # Main frame for the layout
        self.main_frame = tk.Frame(root, padx=10, pady=10)
        self.main_frame.grid(row=0, column=0)
        self.populate_main_screen()
    def populate_main_screen(self) -> None:
        """Populate the main screen with stock selection, date selection, and action buttons.
        This method creates the main interface components including:
        - Stock selection dropdown
        - Year and month selection dropdowns
        - Action buttons for showing calendar and downloading
        - Sentiment display
        """

        # Clear the main frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        # Stock selection
        stock_frame = tk.Frame(self.main_frame, pady=5)
        stock_frame.grid(row=0, column=0, sticky="w")

        tk.Label(stock_frame, text="Select stock:").grid(row=0, column=0, padx=5, pady=5)
        self.stock_entry = ttk.Combobox(
            stock_frame, 
            values=["TSLA", "CLSK", "NVDA", "PLTR", "SPY","BTC-USD"], 
            state="readonly"
        )
        self.stock_entry.grid(row=0, column=1, padx=5)
        self.stock_entry.set("CLSK")
        self.web_scraper_instance = web_scraper(self.stock_entry.get())
        self.navigate_button = tk.Button(
            stock_frame, 
            text="→", 
            font=("Arial", 12), 
            command=self.re_populate_screen
        )
        self.navigate_button.grid(row=0, column=5, padx=2, pady=2)

        # Year and Month selection
        date_frame = tk.Frame(self.main_frame, pady=5)
        date_frame.grid(row=1, column=0, sticky="w")

        tk.Label(date_frame, text="Select Year:").grid(row=0, column=0, padx=5, pady=5)
        self.year_entry = ttk.Combobox(
            date_frame, 
            values=[2021, 2022, 2023, 2024, 2025], 
            state="readonly"
        )
        self.year_entry.grid(row=0, column=1, padx=5)
        self.year_entry.set(datetime.now().year)

        tk.Label(date_frame, text="Select Month:").grid(row=0, column=2, padx=5, pady=5)
        self.month_entry = ttk.Combobox(
            date_frame, 
            values=list(range(1, 13)), 
            state="readonly"
        )
        self.month_entry.grid(row=0, column=3, padx=5)
        self.month_entry.set(datetime.now().month)

        # Buttons for calendar actions
        button_frame = tk.Frame(self.main_frame, pady=10, padx=10)
        button_frame.grid(row=2, column=0, sticky="w")
        button_frame.grid_columnconfigure(1, minsize=120)
        self.show_button = tk.Button(
            button_frame, 
            text="Show Calendar", 
            command=self.show_calendar
        )
        self.show_button.grid(row=1, column=0, padx=5, pady=5)

        self.sentiment_text = tk.Text(button_frame, height=1, width=20)
        self.sentiment_text.grid(row=1, column=2, padx=5, pady=5)
        self.get_sentiment_data()

        action_frame = tk.Frame(self.main_frame, pady=10)
        action_frame.grid(row=3, column=0, sticky="w")
        self.download_button = tk.Button(
            action_frame, 
            text="Download", 
            command=self.download_calendar
        )
        self.download_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.fetch_news_button = tk.Button(
            action_frame, 
            text="Fetch news", 
            command=lambda: self.fetch_news()
        )
        self.fetch_news_button.grid(row=0, column=2, padx=5, pady=5)

        self.crypto_button = tk.Button(
        action_frame,
        text="Crypto Metrics",
        command=self.open_crypto_page
    )
        
        self.crypto_button.grid(row=0, column=3, padx=5, pady=5)

    def open_crypto_page(self) -> None:
        """Open a placeholder window for Crypto Metrics."""
        for widget in self.main_frame.winfo_children():
                widget.destroy()

    def re_populate_screen(self) -> None:
        """Replace the main screen with portfolio details view.
        This method:
        1. Clears the main screen
        2. Calculates portfolio values for each stock
        3. Displays total portfolio value
        4. Creates a pie chart showing portfolio distribution
        """
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        stocks: List[str] = ["TSLA", "NVDA", "CLSK", "DKIGI.CO","CASH"]
        usd_dkk: float = float(self.stock_reader_instance.get_last_trading_day_close(
            datetime.now().year, 
            datetime.now().month,
            "DKK=X"
        ))
        # Portfolio data
        portfolio: Dict[str, Dict[str, float]] = {
            stocks[0]: {
                "shares": 60,
                "price": float(self.stock_reader_instance.get_last_trading_day_close(
                    datetime.now().year,
                    datetime.now().month,
                    stocks[0]
                ))
            },
            stocks[1]: {
                "shares": 155,
                "price": float(self.stock_reader_instance.get_last_trading_day_close(
                    datetime.now().year,
                    datetime.now().month,
                    stocks[1]
                ))
                
            },
            stocks[2]: {
                "shares": 500,
                "price": float(self.stock_reader_instance.get_last_trading_day_close(
                    datetime.now().year,
                    datetime.now().month,
                    stocks[2]
                ))
            },
            stocks[3]: {
                "shares": 993,
                "price": float(self.stock_reader_instance.get_last_trading_day_close(
                    datetime.now().year,
                    datetime.now().month,
                    stocks[3]
                )) / usd_dkk
            }
        }
        dkigi_value = portfolio["DKIGI.CO"]["shares"] * portfolio["DKIGI.CO"]["price"]*usd_dkk
        # Calculate portfolio value
        nordnet_value: float = sum(stock["shares"] * stock["price"] for stock in portfolio.values())*usd_dkk-dkigi_value
        db_value = dkigi_value
        total_value = nordnet_value+db_value

        # Back button (Top-left corner)
        tk.Button(
            self.main_frame,
            text="←",
            font=("Arial", 12),
            command=self.populate_main_screen
        ).grid(row=0, column=0, sticky="w", padx=10, pady=10)
    
        # Display portfolio value
        tk.Label(
            self.main_frame,
            text=f"NordNet Value: {nordnet_value:,.2f}DKK",
            font=("Arial", 16),
            fg="green"
        ).grid(row=3, column=0)
        tk.Label(
            self.main_frame,
            text=f"db_value: {db_value:,.2f}DKK",
            font=("Arial", 16),
            fg="green"
        ).grid(row=4, column=0)
        tk.Label(
            self.main_frame,
            text=f"total value: {total_value:,.2f}DKK",
            font=("Arial", 16),
            fg="green"
        ).grid(row=5, column=0)


        # Prepare data for the pie chart
        labels: List[str] = list(portfolio.keys())
        sizes: List[float] = [stock["shares"] * stock["price"] for stock in portfolio.values()]
        colors: List[str] = ["#FF5733", "#33FF57", "#3357FF", "#FF33A8", "Brown"]

        # Create the pie chart
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors
        )
        ax.axis("equal")
        
        # Embed the chart in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, self.main_frame)
        canvas.get_tk_widget().grid(row=6, column=0, pady=10)
        
        # Close the figure after rendering
        plt.close(fig)

    def get_news_links_for_month(self, year: int, month: int) -> List[str]:
        """Retrieve news links for a specific month and year.
        
        Args:
            year: The year to get news links for
            month: The month to get news links for (1-12)
            
        Returns:
            List of news article URLs for the specified month and year
            
        Raises:
            FileNotFoundError: If the news data file cannot be found
            json.JSONDecodeError: If the news data file is not valid JSON
        """
        try:
            # Load the news data from the JSON file
            with open('json_folder\\article_links.json', 'r', encoding='utf-8') as f:
                news_data: List[Dict[str, str]] = json.load(f)
            
            # List to store links for the specified month and year
            month_links: List[str] = []
            
            # Iterate through each article to filter by year and month
            for article in news_data:
                try:
                    # Parse the date from the JSON into a datetime object
                    article_date = datetime.strptime(article['date'], "%B %d, %Y")
                    
                    # Check if the year and month match the parameters
                    if article_date.year == year and article_date.month == month:
                        month_links.append(article['link'])
                except (KeyError, ValueError) as e:
                    print(f"Error processing article: {e}")
                    continue

            return month_links
            
        except FileNotFoundError:
            print("The news_releases.json file was not found.")
            return []
        except json.JSONDecodeError:
            print("Error decoding the JSON file.")
            return []
        except Exception as e:
            print(f"Unexpected error while getting news links: {e}")
            return []
        
    def open_day_details_window(self, year: int, month: int, day: int, stock: str) -> None:
        """Open a window showing detailed stock data for a specific day.
        Args:
            year: The year of the stock data
            month: The month of the stock data (1-12)
            day: The day of the stock data
            stock: The stock symbol to display data for
        This method:
        1. Registers the Edge browser if available
        2. Creates a new window with stock details
        3. Displays price data, moving averages, and sentiment
        4. Shows news links if available
        """
        # Register Edge browser if available
        if os.path.exists(self.path):
            webbrowser.register("edge", None, webbrowser.BackgroundBrowser(self.path))
        else:
            print("Microsoft Edge not found at default paths.")

        # Get stock data
        btc_mined:float = self.web_scraper_instance.calculate_total_btc(self.web_scraper_instance.bitcoin_data_2024)
        day_window = tk.Toplevel(self.root)
        day_window.title(f"Stock Details for {stock} - {year}-{month:02d}-{day:02d}")
        # Fetch various data points
        today = datetime.now()
        is_today = (today.year == year and today.month == month and today.day == day)
        pre_market_price = get_pre_market_price_ticker(stock)
        if is_today and pre_market_price is not None:
        # ➔ TODAY: Show pre-market data
            tk.Label(day_window, text=f"Today (Pre-Market) - {stock}", font=("Arial", 14)).pack(pady=5)
            tk.Label(day_window, text=f"Pre-Market Price: ${pre_market_price:.2f}", font=("Arial", 12)).pack(pady=5)

        data = self.stock_reader_instance.get_data_for_day(year, month, day, stock)
        start_date = f"{year}-{month}-{day}"
        sentiment_data = self.stock_reader_instance.get_sentiment(stock, start_date, start_date)
        ma20 = self.stock_reader_instance.get_moving_average(
            self.stock_reader_instance.start_date,
            self.stock_reader_instance.end_date,
            stock,
            ma20=True
        )
        #print((datetime.strptime(self.stock_reader_instance.end_date, "%Y-%m-%d") - timedelta(days=200)).strftime("%Y-%m-%d"))
        ma50 = self.stock_reader_instance.get_moving_average(
            self.stock_reader_instance.start_date,
            self.stock_reader_instance.end_date,
            stock,
            ma20=False
        )
        # Get news links if available
        links: List[str] = self.get_news_links_for_month(year, month) if stock == "CLSK" else []

        if isinstance(data, pd.Series):
            # Extract and convert data to standard Python types
            open_price: float = float(data['Open'].iloc[0])
            high_price: float = float(data['High'].iloc[0])
            low_price: float = float(data['Low'].iloc[0])
            close_price: float = float(data['Close'].iloc[0])
            volume: int = int(data['Volume'].iloc[0])
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

            # Display BTC mined for CLSK
            if stock == "CLSK":
                tk.Label(day_window, text=f"BTC mined: {btc_mined:,}", font=("Arial", 12)).pack(pady=2)
                link = tk.Label(
                    day_window,
                    text="Click here to see the total network hashrate",
                    font=("Arial", 10),
                    fg="blue",
                    cursor="hand2"
                )
                link.pack(pady=1)
                link.bind(
                    "<Button-1>",
                    lambda e, url="https://minerstat.com/coin/BTC/network-hashrate": webbrowser.get("edge").open(url)
                )
                
                link = tk.Label(
                    day_window,
                    text="Click here to see the bitcoin mining address",
                    font=("Arial", 10),
                    fg="blue",
                    cursor="hand2"
                )
                link.pack(pady=1)
                link.bind(
                    "<Button-1>",
                    lambda e, url="https://bitref.com/3KmNWUNVGoTzHN8Cyc1kVhR1TSeS6mK9ab": webbrowser.get("edge").open(url)
                )
            # Handle sentiment data
            if sentiment_data is not None:
                sentiment, urls = sentiment_data
                tk.Label(day_window, text=f"Sentiment Score: {sentiment:.2f}", font=("Arial", 12)).pack(pady=2)
                
                if urls:
                    tk.Label(day_window, text="Articles:", font=("Arial", 12)).pack(pady=2)
                    for url in urls:
                        link = tk.Label(
                            day_window,
                            text=url,
                            font=("Arial", 10),
                            fg="blue",
                            cursor="hand2"
                        )
                        link.pack(pady=1)
                        link.bind("<Button-1>", lambda e, url=url: webbrowser.get("edge").open(url))
                for lnk in links:
                    link = tk.Label(
                        day_window,
                        text=lnk,
                        font=("Arial", 10),
                        fg="blue",
                        cursor="hand2"
                    )
                    link.pack(pady=1)
                    link.bind("<Button-1>", lambda e, url=lnk: webbrowser.get("edge").open(url))
    def fetch_news(self) -> None:
        """Fetch news articles and earnings data for the selected stock.
        This method:
        1. Gets the currently selected stock
        2. Scrapes articles if the stock is CLSK
        3. Scrapes earnings data for any stock
        """
        self.stock_entry: str = str(self.stock_entry.get())
        self.web_scraper_instance = web_scraper(self.stock_entry)
        # Only scrape articles for CLSK
        if self.stock_entry == "CLSK":
            self.web_scraper_instance.scrape_articles()
            self.web_scraper_instance.scrape_bitcoin_address()
        # Scrape earnings date for any stock
        self.web_scraper_instance.scrape_earnings()

    def get_sentiment_data(self) -> None:
        """Retrieve and display sentiment data for the current day.
        This method:
        1. Attempts to load sentiment data from a JSON file
        2. If the data is not available or outdated, scrapes new data
        3. Displays the sentiment value in the UI
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        try:
            # Check if the data is in the .json file
            with open('json_folder\\feargreed.json', 'r', encoding='utf-8') as f:
                sentiment_data = json.load(f)
                if sentiment_data[0]["date"] == current_date:
                    self.sentiment_text.insert(tk.END, f'Sentiment value: {sentiment_data[0]["fear_greed_index"]}')
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Could not retrieve the data: {e}")
        if sentiment_data[0]["date"] != current_date:
            sentiment_value = self.web_scraper_instance.scrape_fear_greed_index(
                self.web_scraper_instance.sentiment_url
            )
            
            if sentiment_value:
                self.sentiment_text.insert(tk.END, f'Sentiment value: {sentiment_value}')
            else:
                self.sentiment_text.insert(tk.END, "Could not retrieve sentiment data.")
    def show_calendar(self) -> None:
        """Display the stock calendar for the selected year, month, and stock.
        This method:
        1. Gets the selected year, month, and stock from the UI
        2. Creates and displays the calendar view
        """
        year: int = int(self.year_entry.get())
        month: int = int(self.month_entry.get())
        stock: str = str(self.stock_entry.get())
        # Draw the calendar
        self.stock_reader_instance.create_month_calendar_view(year, month, stock, download=False)

    def download_calendar(self) -> None:
        """Download the stock calendar as a PDF file.
        This method:
        1. Gets the selected year, month, and stock from the UI
        2. Creates the calendar view
        3. Saves it as a PDF file with a descriptive name
        """
        year: int = int(self.year_entry.get())
        month: int = int(self.month_entry.get())
        stock: str = str(self.stock_entry.get())
        self.figure = self.stock_reader_instance.create_month_calendar_view(
            year,
            month,
            stock,
            download=True)
        # Save the figure as a PDF
        filename: str = f"{stock}_{year}_{month}.pdf"
        self.figure.savefig(filename)
    
# Initialize Tkinter
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    # app.get_news_links_for_month(2024,10)
    root.mainloop()
