
from Get_Stock_Data import stock_reader
from datetime import datetime
import tkinter as tk
from tkinter import ttk            
import tkinter as tk
import pandas as pd
import webbrowser
import os
import json
from datetime import datetime
from web_scraper import web_scraper
class App:
    def __init__(self, root):
        self.stock_reader_instance = stock_reader(day_details_callback=self.open_day_details_window)
        self.web_scraper_instance = None
        self.path = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
        self.root = root
        self.root.title("Stock Data Calendar Viewer")
        
        current_date = datetime.now()
        current_month = current_date.month
        current_year = current_date.year

        # Main frame for the layout
        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.grid(row=0, column=0)

        # Stock selection
        stock_frame = tk.Frame(main_frame, pady=5)
        stock_frame.grid(row=0, column=0, sticky="w")
        tk.Label(stock_frame, text="Select stock:").grid(row=0, column=0, padx=5, pady=5)
        self.stock_entry = ttk.Combobox(stock_frame, values=["SPY", "TSLA", "CLSK", "NVDA"], state="readonly")
        self.stock_entry.grid(row=0, column=1, padx=5)
        self.stock_entry.set("CLSK")
        # Year and Month selection
        date_frame = tk.Frame(main_frame, pady=5)
        date_frame.grid(row=1, column=0, sticky="w")
        tk.Label(date_frame, text="Select Year:").grid(row=0, column=0, padx=5, pady=5)
        self.year_entry = ttk.Combobox(date_frame, values=[2022, 2023, 2024,2025], state="readonly")
        self.year_entry.grid(row=0, column=1, padx=5)
        self.year_entry.set(current_year)

        tk.Label(date_frame, text="Select Month:").grid(row=0, column=2, padx=5, pady=5)
        self.month_entry = ttk.Combobox(date_frame, values=list(range(1, 13)), state="readonly")
        self.month_entry.grid(row=0, column=3, padx=5)
        self.month_entry.set(current_month)
        # Buttons for calendar actions
        button_frame = tk.Frame(main_frame, pady=10)
        button_frame.grid(row=2, column=0, sticky="w")
        self.show_button = tk.Button(button_frame, text="Show Calendar", command=self.show_calendar)
        self.show_button.grid(row=0, column=0, padx=5, pady=5)

        self.prev_button = tk.Button(button_frame, text="Previous Month", command=self.prev_month)
        self.prev_button.grid(row=0, column=1, padx=5, pady=5)

        self.next_button = tk.Button(button_frame, text="Next Month", command=self.next_month)
        self.next_button.grid(row=0, column=2, padx=5, pady=5)

        # Download and Fetch news buttons
        action_frame = tk.Frame(main_frame, pady=10)
        action_frame.grid(row=3, column=0, sticky="w")
        self.download_button = tk.Button(action_frame, text="Download", command=self.download_calendar)
        self.download_button.grid(row=0, column=0, padx=5, pady=5)

        self.stock = self.stock_entry.get()
        self.fetch_news_button = tk.Button(action_frame, text="Fetch news", command=lambda: self.fetch_news(self.stock))
        self.fetch_news_button.grid(row=0, column=1, padx=5, pady=5)

    def get_news_links_for_month(self, year, month):
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

        day_window = tk.Toplevel(self.root)
        day_window.title(f"Stock Details for {stock} - {year}-{month:02d}-{day:02d}")
        #     # Add dummy button
        # dummy_button = tk.Button(day_window, text="Dummy Button", command=lambda: print("Dummy button clicked!"))
        # dummy_button.pack(pady=10)
        # Get the data for the selected day
        data = self.stock_reader_instance.get_data_for_day(year, month, day, stock)
        start_date=f"{year}-{month}-{day}"
        sentiment_data = self.stock_reader_instance.get_sentiment(stock, start_date,start_date)
        #print("Sentiment data:", sentiment_data)
        if stock =="CLSK":
            links = self.get_news_links_for_month(year,month)

        else:
            links = []


        if isinstance(data, pd.Series):
            # Extract the required data and convert them to standard Python types
            open_price = float(data['Open'])
            high_price = float(data['High'])
            low_price = float(data['Low'])
            close_price = float(data['Close'])
            volume = int(data['Volume'])
            # Display the data with proper formatting
            tk.Label(day_window, text=f"Date: {year}-{month:02d}-{day:02d}", font=("Arial", 14)).pack(pady=5)
            tk.Label(day_window, text=f"Stock: {stock}", font=("Arial", 14)).pack(pady=5)
            tk.Label(day_window, text=f"Open Price: ${open_price:,.2f}", font=("Arial", 12)).pack(pady=2)
            tk.Label(day_window, text=f"High Price: ${high_price:,.2f}", font=("Arial", 12)).pack(pady=2)
            tk.Label(day_window, text=f"Low Price: ${low_price:,.2f}", font=("Arial", 12)).pack(pady=2)
            tk.Label(day_window, text=f"Close Price: ${close_price:,.2f}", font=("Arial", 12)).pack(pady=2)
            tk.Label(day_window, text=f"Volume: {volume:,}", font=("Arial", 12)).pack(pady=2)
            

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

            else:
                if stock=="CLSK":
                    for lnk in links:
                        link = tk.Label(day_window, text=lnk, font=("Arial", 10), fg="blue", cursor="hand2")
                        link.pack(pady=1)
                        link.bind("<Button-1>", lambda e, url=lnk: webbrowser.get("edge").open(url))
    def fetch_news(self,stock):
         # Fetch the selected stock dynamically from the dropdown
        stock = str(self.stock_entry.get())

        self.web_scraper_instance = web_scraper(stock)

        # Only scrape articles for CLSK
        if stock == "CLSK":
            self.web_scraper_instance.scrape_articles()

        # Scrape earnings for any stock
        self.web_scraper_instance.scrape_earnings()

    def update_calendar(self):
        """Updates the year and month fields in the dropdowns."""
        self.year_entry.set(self.current_year)
        self.month_entry.set(self.current_month)
        self.show_calendar()  # Refresh the calendar view

    def prev_month(self):
        """Move to the previous month."""
        if self.current_month == 1:
            self.current_month = 12
            self.current_year -= 1
        else:
            self.current_month -= 1
        self.update_calendar()

    def next_month(self):
        """Move to the next month."""
        if self.current_month == 12:
            self.current_month = 1
            self.current_year += 1
        else:
            self.current_month += 1
        self.update_calendar()

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

