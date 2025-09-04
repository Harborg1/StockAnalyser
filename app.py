from typing import Dict, List, Optional, Tuple, Any
from stocks.Get_Stock_Data import stock_reader
from stocks.Get_Crypto_Data import crypto_reader
from datetime import datetime
import tkinter as tk
from tkinter import ttk            
import tkinter as tk
import pandas as pd
import webbrowser
import os
import json
from datetime import datetime
from webscrapers.web_scraper import web_scraper
from auxillary.pre_market import get_pre_market_price_ticker
import matplotlib.pyplot as plt
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import EngFormatter

class App:
    """Main application class for the Stock Data Calendar Viewer.
    This class handles the GUI and business logic for viewing and analyzing stock data.
    It provides functionality for viewing stock calendars, portfolio details, and news data.
    """
    def __init__(self, root: ctk.CTk) -> None:
        """Initialize the application.
        Args:
            root: The main Tkinter window instance.
        """
        # Define color scheme
        self.colors = {
            'primary': '#2c3e50',    # Dark blue-gray
            'secondary': '#3498db',   # Bright blue
            'accent': '#e74c3c',      # Red
            'background': '#ecf0f1',  # Light gray
            'text': '#2c3e50',        # Dark blue-gray
            'button': '#3498db',      # Bright blue
            'button_hover': '#2980b9', # Darker blue for hover
            'button_text': 'white'
        }

        # Configure root window
        self.root = root
        self.root.title("Market Metrics Explorer")
        self.root.configure(bg=self.colors['background'])
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure('TFrame', background=self.colors['background'])
        self.style.configure('TLabel', 
                           background=self.colors['background'],
                           foreground=self.colors['text'],
                           font=('Helvetica', 10))
        self.style.configure('TButton',
                           background=self.colors['button'],
                           foreground=self.colors['button_text'],
                           font=('Helvetica', 10, 'bold'),
                           padding=10)
        self.style.configure('TCombobox',
                           background=self.colors['background'],
                           fieldbackground='white',
                           font=('Helvetica', 10))
        
        # Initialize instances
        self.stock_reader_instance = stock_reader(day_details_callback=self.open_day_details_window)
        self.crypto_reader_instance = crypto_reader()
        self.path = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
        
        # Main frame with padding
        self.main_frame = ctk.CTkFrame(root, corner_radius=15, fg_color=self.colors['background'])
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # Configure grid weights
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        self.populate_main_screen()

    
    def cleanup_canvas(self):
        if hasattr(self, 'canvas'):
            try:
                self.canvas.get_tk_widget().destroy()
                plt.close('all')  # Optional but helpful
            except Exception:
                pass

    def create_styled_button(self, parent, text, command, width=150):
        """Create a rounded button using CTkButton from customtkinter."""
        btn = ctk.CTkButton(
            parent,
            text=text,
            command=command,
            width=width,
            height=36,
            corner_radius=12,
            fg_color=self.colors['button'],
            hover_color=self.colors['button_hover'],
            text_color=self.colors['button_text'],
            font=ctk.CTkFont(family='Helvetica', size=12, weight='bold')
        )
        return btn

    def populate_main_screen(self) -> None:
        self.cleanup_canvas()
        """Populate the main screen with styled components."""
        # Clear the main frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Title
        title_label = ctk.CTkLabel(
            self.main_frame,
            text="Market Metrics Explorer",
            font=ctk.CTkFont(family='Helvetica', size=16, weight='bold'),
            text_color=self.colors['primary'],
            fg_color=self.colors['background']
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Stock selection frame
        stock_frame = ttk.Frame(self.main_frame)
        stock_frame.grid(row=1, column=0, sticky="w", pady=10)
        
        ttk.Label(stock_frame, text="Select stock:", font=('Helvetica', 11)).grid(row=0, column=0, padx=5)
        self.stock_entry = ttk.Combobox(
            stock_frame,
            values=["TSLA", "CLSK", "NOVO-B.CO", "NVDA", "PLTR", "SPY", "BTC-USD", "^VIX"],
            state="readonly",
            width=15,
            font=('Helvetica', 11)
        )
        self.stock_entry.grid(row=0, column=1, padx=5)
        self.stock_entry.set("CLSK")
        
        self.web_scraper_instance = web_scraper(self.stock_entry.get())
        self.navigate_button = self.create_styled_button(
            stock_frame, "→", self.re_populate_screen, width=3
        )

        self.navigate_button.grid(row=0, column=2, padx=5)

        # Date selection frame
        date_frame = ttk.Frame(self.main_frame)
        date_frame.grid(row=2, column=0, sticky="w", pady=10)
        
        ttk.Label(date_frame, text="Select Year:", font=('Helvetica', 11)).grid(row=0, column=0, padx=5)
        self.year_entry = ttk.Combobox(
            date_frame,
            values=[2021, 2022, 2023, 2024, 2025],
            state="readonly",
            width=8,
            font=('Helvetica', 11)
        )
        self.year_entry.grid(row=0, column=1, padx=5)
        self.year_entry.set(datetime.now().year)

        ttk.Label(date_frame, text="Select Month:", font=('Helvetica', 11)).grid(row=0, column=2, padx=5)
        self.month_entry = ttk.Combobox(
            date_frame,
            values=list(range(1, 13)),
            state="readonly",
            width=5,
            font=('Helvetica', 11)
        )
        self.month_entry.grid(row=0, column=3, padx=5)
        self.month_entry.set(datetime.now().month)

        # Button to show market activity
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=3, column=0, sticky="w", pady=10)
        
        self.show_button = self.create_styled_button(
            button_frame,
            "Show Market Activity",
            self.show_calendar
        )
        self.show_button.grid(row=0, column=0, padx=5)

        # Combined info frame for sentiment and market state
        info_frame = ttk.Frame(self.main_frame)
        info_frame.grid(row=4, column=0, sticky="w", padx=10, pady=(0, 5))

        self.sentiment_label = tk.Label(
            info_frame,
            text="Sentiment value: --",
            font=('Helvetica', 11),
            bg=self.colors['background'],
            fg=self.colors['text'],
            anchor="w",
            justify="left"
        )
        self.sentiment_label.grid(row=0, column=0, sticky="w", padx=(0, 15))

        self.market_state_label = tk.Label(
            info_frame,
            text="SPY market state: --",
            font=('Helvetica', 11),
            bg=self.colors['background'],
            fg=self.colors['text'],
            anchor="w",
            justify="left"
        )
        self.market_state_label.grid(row=1, column=0, sticky="w")

        self.show_market_state()

        # Action buttons
        action_frame = ttk.Frame(self.main_frame)
        action_frame.grid(row=5, column=0, sticky="w", pady=10)

        self.download_button = self.create_styled_button(
            action_frame, "Download", self.download_calendar
        )
        self.download_button.grid(row=0, column=0, padx=5)

        self.fetch_news_button = self.create_styled_button(
            action_frame, "Fetch News", self.fetch_news
        )
        self.fetch_news_button.grid(row=0, column=1, padx=5)

        self.crypto_button = self.create_styled_button(
            action_frame, "Crypto Metrics", self.open_crypto_page
        )
        self.crypto_button.grid(row=0, column=2, padx=5)


    def open_crypto_page(self) -> None:

        self.cleanup_canvas()
        """Displays different bitcoin data including price, moving average and BTC available on global exchanges """
        # Clear frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Back button
        back_button = self.create_styled_button(
            self.main_frame, "←", self.populate_main_screen, width=40
        )
        back_button.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        # Title
        title_label = ctk.CTkLabel(
            self.main_frame,
            text="Bitcoin Metrics",
            font=ctk.CTkFont(family='Helvetica', size=16, weight='bold'),
            text_color=self.colors['primary'],
            fg_color=self.colors['background']
        )
        title_label.grid(row=1, column=0, columnspan=2, pady=(20, 10))

        # Get metrics
        stock = "BTC-USD"
        current_price = self.crypto_reader_instance.get_last_trading_day_close(
            datetime.now().year, datetime.now().month, stock
        )
        ma20 = self.crypto_reader_instance.get_moving_average(
            self.crypto_reader_instance.start_date, self.crypto_reader_instance.end_date, stock, 20
        )
        ma50 = self.crypto_reader_instance.get_moving_average(
            self.crypto_reader_instance.start_date, self.crypto_reader_instance.end_date, stock, 50
        )

        # Labels
        ctk.CTkLabel(self.main_frame, text=f"BTC Price: ${current_price}", font=ctk.CTkFont(size=14),
                    text_color=self.colors['secondary'], fg_color=self.colors['background']
        ).grid(row=2, column=0, columnspan=2, pady=5)

        ctk.CTkLabel(self.main_frame, text=f"20-day MA: {ma20}", font=ctk.CTkFont(size=14),
                    text_color=self.colors['secondary'], fg_color=self.colors['background']
        ).grid(row=3, column=0, columnspan=2, pady=5)

        ctk.CTkLabel(self.main_frame, text=f"50-day MA: {ma50}", font=ctk.CTkFont(size=14),
                    text_color=self.colors['secondary'], fg_color=self.colors['background']
        ).grid(row=4, column=0, columnspan=2, pady=5)

        json_path = os.path.join("json_folder", "coinglass_balance_24h_change.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        
        data.sort(key=lambda x: x["timestamp"])  # Ensure ascending order

        # Process data
        full_timestamps = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") for entry in data]
        dates = [ts.date() for ts in full_timestamps]
        total_bitcoin = [float(entry["Total bitcoin"].replace(",", "")) for entry in data]

        # Get the data for the last 10 days
        data_10d =total_bitcoin[-10:]
        # Get the dates for the last 10 days
        dates_10d = dates[-10:]

        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)

        btc_changes =  [data_10d[0]-total_bitcoin[-11] if i == 0 else data_10d[i] - data_10d[i - 1] for i in range(len(data_10d))]
        bar_colors = ['green' if change >= 0 else 'red' for change in btc_changes]
        bars = ax.bar(dates_10d, data_10d, color=bar_colors, alpha=0.8, width=0.4)

        btc_formatter = EngFormatter(unit="", places=3)
        ax.yaxis.set_major_formatter(btc_formatter)

        for (bar, change) in (zip(bars, btc_changes)):
            sign = "+" if change >= 0 else ""
            short_change = f"{sign}{round(change/1000):,}K" if abs(change) >= 1000 else f"{sign}{change:,.0f}"
            ax.annotate(short_change, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha='center', fontsize=6.3)

        ax.set_title("Total BTC Held on Exchanges", fontsize=12)
        ax.set_ylabel("Total BTC")
        ax.tick_params(axis='x', labelsize=8, rotation=45)
        ax.set_xticks(dates_10d)
        ax.set_xticklabels([d.strftime('%b %d') for d in dates_10d])
        ax.grid(True, linestyle='--', alpha=0.5)
        min_btc = min(total_bitcoin)*0.999
        max_btc = max(total_bitcoin)*1.011
        padding = (max_btc - min_btc) * 0.01  # 1% padding

        ax.set_ylim(min_btc - padding, max_btc + padding)
        fig.subplots_adjust(left=0.2)
        fig.tight_layout()

        # Render in UI
        self.canvas = FigureCanvasTkAgg(fig, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=5, column=0, columnspan=2, pady=20)



    def re_populate_screen(self) -> None:
        """Replace the main screen with portfolio details view.
        This method:
        1. Clears the main screen
        2. Calculates portfolio values for each stock
        3. Displays total portfolio value
        4. Creates a pie chart showing portfolio distribution
        """

        self.cleanup_canvas()
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Title
        title_label = ctk.CTkLabel(
            self.main_frame,
            text="Portfolio Overview",
            font=ctk.CTkFont(family='Helvetica', size=16, weight='bold'),
            text_color=self.colors['primary'],
            fg_color=self.colors['background']
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Back button (now styled)
        back_button = self.create_styled_button(
            self.main_frame,
            "←",
            self.populate_main_screen,
            width=40
        )
        back_button.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        
        stocks: List[str] = ["TSLA", "NVDA", "CLSK", "DKIGI.CO", "CASH"]
        usd_dkk: float = float(self.stock_reader_instance.get_last_trading_day_close(
            datetime.now().year,
            datetime.now().month,
            "DKK=X"
        ))
        # Portfolio data
        portfolio: Dict[str, Dict[str, float]] = {
            stocks[0]: {
                "shares":62,
                "price": float(self.stock_reader_instance.get_last_trading_day_close(
                    datetime.now().year,
                    datetime.now().month,
                    stocks[0]
                ))
            },
            stocks[1]: {
                "shares": 132,
                "price": float(self.stock_reader_instance.get_last_trading_day_close(
                    datetime.now().year,
                    datetime.now().month,
                    stocks[1]
                ))
                
            },
            stocks[2]: {
                "shares": 600,
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
            },

             stocks[4]: {
                "shares": 1,
                "price": 980/usd_dkk
            }
            }
        
        # Prepare data for the pie chart
        labels: List[str] = ["TSLA", "NVDA", "CLSK", "ETF", "CASH"]
        sizes: List[float] = [stock["shares"] * stock["price"] for stock in portfolio.values()]
        # Define a larger color palette
        colors = [
            '#FF5733', # red-orange
            '#33FF57', # green
            '#3357FF', # blue
            '#FF33A1', # pink
            '#FFD433', # yellow
            '#8E44AD', # purple
            '#1ABC9C', # teal
            '#E67E22', # orange
            '#2ECC71', # emerald green
            '#3498DB'  # sky blue
        ]
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
        

    # Create styled links
    def create_link(self,parent, text, url):
            link = tk.Label(
                parent,
                text=text,
                font=('Helvetica', 10),
                fg=self.colors['secondary'],
                bg=self.colors['background'],
                cursor="hand2"
            )
            link.pack(pady=2)
            link.bind("<Button-1>", lambda e: webbrowser.get("edge").open(url))
            return link
        
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
        if os.path.exists(self.path):
            webbrowser.register("edge", None, webbrowser.BackgroundBrowser(self.path))
        else:
            print("Microsoft Edge not found at default paths.")

        btc_mined: float = self.web_scraper_instance.calculate_total_btc(self.web_scraper_instance.bitcoin_data_2024)
        day_window = tk.Toplevel(self.root)
        day_window.title(f"Stock Details for {stock} - {year}-{month:02d}-{day:02d}")
        day_window.configure(bg=self.colors['background'])
        
        # Add padding to the window
        main_frame = ttk.Frame(day_window, padding="20 20 20 20")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Title
        title_label = tk.Label(
            main_frame,
            text=f"Stock Details for {stock}",
            font=('Helvetica', 16, 'bold'),
            bg=self.colors['background'],
            fg=self.colors['primary'],
            pady=10
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        today = datetime.now()
        is_today = (today.year == year and today.month == month and today.day == day)
        pre_market_price = get_pre_market_price_ticker(stock)
        
        data = self.stock_reader_instance.get_data_for_day(year, month, day, stock)
        if is_today and pre_market_price is not None:
            if data is None:
                pre_market_frame = ttk.Frame(main_frame)
                pre_market_frame.grid(row=1, column=0, columnspan=2, pady=10)
                tk.Label(
                    pre_market_frame,
                    text="Today (Pre-Market)",
                    font=('Helvetica', 12, 'bold'),
                    bg=self.colors['background'],
                    fg=self.colors['accent']
                ).pack(pady=5)
                tk.Label(
                    pre_market_frame,
                    text=f"Pre-Market Price: ${pre_market_price:.2f}",
                    font=('Helvetica', 11),
                    bg=self.colors['background'],
                    fg=self.colors['text']
                ).pack(pady=5)

        start_date = f"{year}-{month}-{day}"
        sentiment_data = self.stock_reader_instance.get_sentiment(stock, start_date, start_date)
        ma20 = self.stock_reader_instance.get_moving_average(
            self.stock_reader_instance.start_date,
            self.stock_reader_instance.end_date,
            stock,
            20
        )
        ma50 = self.stock_reader_instance.get_moving_average(
            self.stock_reader_instance.start_date,
            self.stock_reader_instance.end_date,
            stock,
            50
        )
        links: List[str] = self.get_news_links_for_month(year, month) if stock == "CLSK" else []

        if isinstance(data, pd.Series):
            # Create a frame for price data
            price_frame = ttk.Frame(main_frame)
            price_frame.grid(row=2, column=0, columnspan=2, pady=10)
            
            # Extract and convert data
            open_price: float = float(data['Open'].iloc[0])
            high_price: float = float(data['High'].iloc[0])
            low_price: float = float(data['Low'].iloc[0])
            close_price: float = float(data['Close'].iloc[0])
            volume: int = int(data['Volume'].iloc[0])
            
            # Display price data with styling
            price_data = [
                ("Open Price", f"${open_price:,.2f}"),
                ("High Price", f"${high_price:,.2f}"),
                ("Low Price", f"${low_price:,.2f}"),
                ("Close Price", f"${close_price:,.2f}"),
                ("Volume", f"{volume:,}"),
                ("20-day MA", f"{ma20:,.2f}"),
                ("50-day MA", f"{ma50:,.2f}")
            ]
            
            for i, (label, value) in enumerate(price_data):
                tk.Label(
                    price_frame,
                    text=label,
                    font=('Helvetica', 11),
                    bg=self.colors['background'],
                    fg=self.colors['text']
                ).grid(row=i, column=0, sticky="w", padx=5, pady=2)
                
                tk.Label(
                    price_frame,
                    text=value,
                    font=('Helvetica', 11, 'bold'),
                    bg=self.colors['background'],
                    fg=self.colors['primary']
                ).grid(row=i, column=1, sticky="w", padx=5, pady=2)

            # Display BTC mined for CLSK
            if stock == "CLSK":
                btc_frame = ttk.Frame(main_frame)
                btc_frame.grid(row=3, column=0, columnspan=2, pady=10)
                
                tk.Label(
                    btc_frame,
                    text=f"BTC mined: {btc_mined:,}",
                    font=('Helvetica', 11),
                    bg=self.colors['background'],
                    fg=self.colors['text']
                ).pack(pady=5)
                
               
                self.create_link(btc_frame, "View Network Hashrate", "https://minerstat.com/coin/BTC/network-hashrate")
                self.create_link(btc_frame, "View Bitcoin Mining Address", "https://bitref.com/3KmNWUNVGoTzHN8Cyc1kVhR1TSeS6mK9ab")

            # Handle sentiment data
            if sentiment_data is not None:
                sentiment, urls = sentiment_data
                sentiment_frame = ttk.Frame(main_frame)
                sentiment_frame.grid(row=4, column=0, columnspan=2, pady=10)
                
                tk.Label(
                    sentiment_frame,
                    text=f"Sentiment Score: {sentiment:.2f}",
                    font=('Helvetica', 11),
                    bg=self.colors['background'],
                    fg=self.colors['text']
                ).pack(pady=5)

                if urls:
                    tk.Label(
                        sentiment_frame,
                        text="Related Articles:",
                        font=('Helvetica', 11, 'bold'),
                        bg=self.colors['background'],
                        fg=self.colors['primary']
                    ).pack(pady=5)

                    for url in urls:
                        self.create_link(sentiment_frame, url, url)

                for lnk in links:
                    self.create_link(sentiment_frame, lnk, lnk)

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
        filename: str = f"images/{stock}_{year}_{month}.pdf"
        self.figure.savefig(filename)


    def show_market_state(self) -> None:
        current_date = datetime.now().strftime("%Y-%m-%d")
        try:
            # Check if the data is in the .json file
            with open('json_folder\\feargreed.json', 'r', encoding='utf-8') as f:
                sentiment_data = json.load(f)
                if sentiment_data[-1]["date"] == current_date:
                    self.sentiment_label.config(text = f'Sentiment value: {sentiment_data[-1]["fear_greed_index"]}')
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Could not retrieve the data: {e}")
        if sentiment_data[-1]["date"] != current_date:
            sentiment_value = self.web_scraper_instance.scrape_fear_greed_index(self.web_scraper_instance.sentiment_url)
            if sentiment_value:
                self.sentiment_label.config(text =  f'Sentiment value: {sentiment_value}')
            else:
                self.sentiment_label.config(text = "Could not retrieve sentiment data.")

        # Update market state
        try:
            diff = self.stock_reader_instance.get_spy_distance_from_ath()
            if isinstance(diff, str):
                state_text = f"SPY: {diff}"
            else:
                if diff >= -10:
                    label = "🐂  Bull Market"
                elif -20 < diff < -10:
                    label = "⚠️ Correction"
                elif -30 < diff <= -20:
                    label = "🐻 Bear Market"
                else:
                    label = "📉 Recession"
                state_text = f"SPY is {diff:.2f}% from ATH: {label}"
        except Exception as e:
            state_text = f"SPY error: {e}"
        self.market_state_label.config(text=state_text)

# Initialize Tkinter
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    def on_closing():
        try:
            if hasattr(app, 'canvas'):
                app.canvas.get_tk_widget().destroy()
                app.canvas.get_tk_widget().after_cancel(app.canvas._idle_callback_id) if hasattr(app.canvas, '_idle_callback_id') else None
            plt.close('all')
        except Exception as e:
            print(f"Cleanup error: {e}")
        finally:
            try:
                root.quit()     # Stop the mainloop
                root.destroy()  # Destroy the window
            except Exception as e:
                print(f"Final shutdown error: {e}")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
    