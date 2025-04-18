# earnings_scraper.py

import json
from collections import defaultdict
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import datetime
import time


from selenium.webdriver.support import expected_conditions as EC

class TextPresentInElement(object):
    def __init__(self, locator):
        self.locator = locator
    def __call__(self, driver):
        try:
            element = driver.find_element(*self.locator)
            return element.text.strip() != ''
        except:
            return False
        


# https://googlechromelabs.github.io/chrome-for-testing/#stable
class web_scraper:
    def __init__(self, stock_name):
        """Initialize with the stock name (ticker)."""
        self.stock_name = stock_name.upper()
        self.earnings_url = f'https://www.nasdaq.com/market-activity/stocks/{self.stock_name}/earnings'
        self.cpi_url = "https://www.bls.gov/schedule/news_release/cpi.htm"
        self.sentiment_url = "https://edition.cnn.com/markets/fear-and-greed"
    
        self.json_file_path_earnings = "json_folder\\stock_earnings.json"
        self.json_file_path_cpi = "json_folder\\cpi.json"
        self.json_file_path_fear_greed = "json_folder\\feargreed.json"
        self.driver = None
        self.bitcoin_data = "json_folder\\bitcoin_address_data_all_time.json"
        self.bitcoin_data_2024 = "json_folder\\bitcoin_address_data_2024.json"

    def setup_driver(self):
        """Sets up the headless Chrome driver."""
        options = Options()
        options.headless = True
        service = Service(executable_path="chromedriver.exe")  # Update with your chromedriver path
        self.driver = webdriver.Chrome(service=service, options=options)
        return self.driver
    def scrape_cpi(self):
        self.setup_driver()
        self.driver.get(self.cpi_url)

        existing_cpi = []
        if os.path.exists(self.json_file_path_cpi):
            with open(self.json_file_path_cpi, "r", encoding="utf-8") as file:
                existing_cpi = json.load(file)

        # Wait for the CPI elements to load
        try:
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "release-list-odd-row"))
                or EC.presence_of_element_located((By.CLASS_NAME, "release-list-even-row"))
            )
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        except Exception as e:
            print(f"Error loading CPI page: {e}")
            self.driver.quit()
            return []
        
        # Select rows with CPI data from both odd and even rows
        cpi_rows = soup.select(".release-list-odd-row, .release-list-even-row")
        new_cpi = []

        # Extract only the date from the second <td> element
        for row in cpi_rows:
            # Get all <td> elements in the row
            columns = row.find_all("td")
            if len(columns) >= 2:  # Ensure there are at least two columns
                date_text = columns[1].get_text(strip=True)  # Extract the second column's text
                # Skip if this date is already in existing_cpi
                if any(item['date'] == date_text for item in existing_cpi):
                    continue

                # Append the extracted date to the new CPI list
                new_cpi.append({
                    "date": date_text
                })

        self.driver.quit()
        # Save new CPI data if any found
        if new_cpi:
            with open(self.json_file_path_cpi, "w", encoding="utf-8") as file:
                json.dump(existing_cpi + new_cpi, file, indent=4, ensure_ascii=False)
            print(f"Added {len(new_cpi)} new CPI date(s)")
        else:
            print("No new CPI dates found")

        return new_cpi

    def scrape_fear_greed_index(self,url):
        if self.driver is None:
            self.setup_driver()
        self.driver.get(url)

        existing_values = []
        if os.path.exists(self.json_file_path_fear_greed):
            with open(self.json_file_path_fear_greed, "r", encoding="utf-8") as file:
                existing_values = json.load(file)
        try:
            time.sleep(6)
            locator = (By.CLASS_NAME, "market-fng-gauge__dial-number-value")
            WebDriverWait(self.driver, 1500).until(EC.presence_of_element_located(locator))
            elements = self.driver.find_elements(By.CLASS_NAME, "market-fng-gauge__dial-number-value")
            for el in elements:
                value = el.text.strip()
                if value:
                    data = []
                    #Save to JSON file
                    data.append({
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "fear_greed_index": value
                    })
                    if data[0] ["date"] ==existing_values[0]["date"]:
                        break
                    with open(self.json_file_path_fear_greed, "w") as json_file:
                        json.dump(data+existing_values, json_file, indent=4)
                    self.driver.quit()
                    return value
            return None
        
        except Exception as e:
            print(f"Error scraping Fear & Greed index: {e}")
            return None
        
    def scrape_earnings(self):
        """Scrapes earnings dates, updates changes, and saves them to a JSON file."""
        self.setup_driver()
        self.driver.get(self.earnings_url)
        # Load existing earnings data if available
        existing_earnings = []
        if os.path.exists(self.json_file_path_earnings):
            with open(self.json_file_path_earnings, "r", encoding="utf-8") as file:
                existing_earnings = json.load(file)

        # Wait for the earnings elements to load
        try:
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "announcement-date"))
            )
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        except Exception as e:
            print(f"Error loading earnings page: {e}")
            self.driver.quit()
            return []

        # Extract new earnings dates
        earnings_elements = soup.select(".announcement-date")
        new_earnings = []
        updated_earnings = []

        for element in earnings_elements:
            date_text = element.get_text(strip=True)

            # Check if the stock already has an earnings date recorded
            existing_entry = next((item for item in existing_earnings if item['stock'] == self.stock_name), None)
            
            if existing_entry:
                # If the date has changed, remove the old entry
                if existing_entry['date'] != date_text:
                    print(f"Earnings date changed for {self.stock_name}: {existing_entry['date']} -> {date_text}")
                    existing_earnings = [item for item in existing_earnings if not (item['stock'] == self.stock_name)]
                    updated_earnings.append({"date": date_text, "stock": self.stock_name})
            else:
                new_earnings.append({"date": date_text, "stock": self.stock_name})

        self.driver.quit()


        # Merge and save the updated earnings data
        final_earnings = existing_earnings + updated_earnings + new_earnings
        with open(self.json_file_path_earnings, "w", encoding="utf-8") as file:
            json.dump(final_earnings, file, indent=4, ensure_ascii=False)

        print(f"Updated {len(updated_earnings)} earnings date(s) and added {len(new_earnings)} new earnings date(s) for {self.stock_name}")
        return updated_earnings + new_earnings
    
    def scrape_articles(self):

        """Function to scrape articles and return new links."""
        driver = self.setup_driver()

        url = 'https://investors.cleanspark.com/news/'
        driver.get(url)

        json_file_path = "json_folder\\article_links.json"
        existing_links = []

        if os.path.exists(json_file_path):
            with open(json_file_path, "r", encoding="utf-8") as file:
                existing_links = json.load(file)

        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/news/news-details/']"))
            )
            soup = BeautifulSoup(driver.page_source, 'html.parser')
        finally:
            pass
        
        articles = soup.select("a[href*='/news/news-details/']")
        new_links = []

        for article in articles:
            title = article.get_text(strip=True)
            link = article['href']
            if not link.startswith("http"):
                link = "https://investors.cleanspark.com" + link
                
            if any(item['link'] == link for item in existing_links):
                continue

            driver.get(link)
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "_ctrl0_ctl49_spanDate"))
                )
                article_soup = BeautifulSoup(driver.page_source, 'html.parser')
                date_element = article_soup.find("span", id="_ctrl0_ctl49_spanDate")
                article_date = date_element.get_text(strip=True) if date_element else "Date not found"
            except Exception as e:
                article_date = f"Error retrieving date: {e}"

            new_links.append({
                "title": title,
                "link": link,
                "date": article_date
            })

        driver.quit()

        if new_links:
            with open(json_file_path, "w", encoding="utf-8") as file:
                json.dump(existing_links + new_links, file, indent=4, ensure_ascii=False)
            print(f"Added {len(new_links)} new article(s)")
        else:

            print("No recent news")

        return new_links
    

    def scrape_bitcoin_address(self):
        """
        Scrapes data from the bitcoin address of CLSK that is more recent than the cutoff date."""

        # Calculate target_count as the number of days from today to December 4th
        today = datetime.now()
        cutoff_date = datetime(2024, 12, 1)

        days_difference = (today - cutoff_date).days+1
        target_count = days_difference
        print(f"Target count (days difference): {target_count}")

        url = "https://bitref.com/3KmNWUNVGoTzHN8Cyc1kVhR1TSeS6mK9ab"
        json_file_path = self.bitcoin_data_2024
        self.setup_driver()
        self.driver.get(url)
        existing_data = []
            # Delete the existing JSON file if it exists
        if os.path.exists(json_file_path):
            os.remove(json_file_path)
            print(f"Deleted existing file: {json_file_path}")
                    
        total_clicks = 0
        while len(existing_data) < target_count:
            # Click "Load More Transactions" to load more data
            try:
                load_more_button = WebDriverWait(self.driver, 100).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='button'][class*='btn-outline-secondary'][onclick*='getTransactions']"))
                )
                
                self.driver.execute_script("arguments[0].click();", load_more_button)
                WebDriverWait(self.driver, 100).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#txs tr:not(#loading)"))
                )
                total_clicks += 1
            except Exception as e:
                print(f"Error clicking the button or loading more data: {e}")
                break

            # Parse the current page to get new data
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            table_body = soup.find("tbody", id="txs")
            if not table_body:
                print("No table body with ID 'txs' found")
                break
                
            rows = table_body.find_all("tr")
            new_data = []

            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 4:  # Ensure there are enough cells in the row
                    continue
                date_text = cells[1].get_text(strip=True)

                if date_text == 'Unconfirmed!🚀': continue
                row_date = datetime.strptime(date_text.split()[0], "%Y-%m-%d")  # Parse the date 
                btc_mined = cells[3].get_text(strip=True)
                value_class = cells[3].get("class", [])
                # Skip the value if it is before the cutoff date
                if row_date < cutoff_date:
                    continue
                # We skip the negative btc mined amounts.
                if "text-danger" in value_class:
                    continue

                # Skip if the data is already in the existing data
                if any(item['btc_mined'] == btc_mined for item in existing_data):
                    continue

                # Append new data
                new_data.append({"date": str(date_text),"btc_mined": btc_mined})

            # Add new data to the existing list
            existing_data.extend(new_data)

            # Save the updated data
            with open(json_file_path, "w", encoding="utf-8") as file:
                json.dump(existing_data, file, indent=4, ensure_ascii=False)

            # If the target count is reached, stop
            if len(existing_data) >= target_count:
                print(f"Target count of {target_count} data points reached.")
                break
        print(f"Total clicks performed: {total_clicks}")
        self.driver.quit()
        return existing_data
    

    def scrape_bitcoin_address_all_time(self):
        """Scrapes data from the specified Bitcoin address page, including all available transactions."""
        url = "https://bitref.com/3KmNWUNVGoTzHN8Cyc1kVhR1TSeS6mK9ab"
        json_file_path = self.bitcoin_data
        
        # Setup driver
        self.setup_driver()
        self.driver.get(url)
        existing_data = []
        if os.path.exists(json_file_path):
            os.remove(json_file_path)

        new_data = []
        while True:
            try:
                # Click the "Load More Transactions" button
                load_more_button = WebDriverWait(self.driver, 50).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='button'][class*='btn-outline-secondary'][onclick*='getTransactions']"))
                )
                self.driver.execute_script("arguments[0].click();", load_more_button)
                WebDriverWait(self.driver, 50).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#txs tr:not(#loading)"))
                )
            except Exception:
                # Break the loop if the button is not found or not clickable
                print("Button was not found or no more data to load.")
                break

        # Parse the current page after each button click
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        table_body = soup.find("tbody", id="txs")
        if not table_body:
            print("No table body with ID 'txs' found")
            return
        
        rows = table_body.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 4:  # Ensure there are enough cells in the row
                continue
            date_text = cells[1].get_text(strip=True)
            btc_mined = cells[3].get_text(strip=True)
            value_class = cells[3].get("class", [])

            if date_text=="":
                continue
             # Skip negative btc mined amounts
             # Skip if the data is already in the existing data
            if any(item['btc_mined'] == btc_mined for item in existing_data):
                continue

            if "text-danger" in value_class:
                continue
            # Append new data
            new_data.append({"date": date_text, "btc_mined": btc_mined})

        # Add new data to the existing list
        existing_data.extend(new_data)

        # Save the updated data
        with open(json_file_path, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, indent=4, ensure_ascii=False)

        self.driver.quit()
        return existing_data
    
    def calculate_total_btc(self,json_data):
         # Load Bitcoin data from the file
        nov_30_2024 = 9297
        # Source: https://investors.cleanspark.com/news/news-details/2024/CleanSpark-Releases-November-2024-Bitcoin-Mining-Update/default.aspx
        bitcoin_data = []
        if os.path.exists(json_data):
            with open(json_data, "r", encoding="utf-8") as file:
                bitcoin_data = json.load(file)

        total_sum = nov_30_2024

        for item in bitcoin_data:
            # Convert the 'data' string to a float and add it to the total
            try:
                value = float(item['btc_mined'].replace('+', ''))
                total_sum += value
            except ValueError:
                print(f"Skipping invalid value: {item['btc_mined']}")
        return total_sum
    
    def calculate_btc_mined_per_month(self, json_file_path):

        if os.path.exists(json_file_path):
            with open(json_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

        btc_by_month = defaultdict(float)
        for entry in data:
        # Extract month from the date
            date = datetime.strptime(entry["date"], "%Y-%m-%d %H:%M:%S")
            month = date.strftime("%Y-%m")  # Format as "YYYY-MM"
            btc_mined = float(entry["btc_mined"])  # Convert btc_mined to float
            btc_by_month[month] += btc_mined
        # Print the results
        print("Bitcoin mined per month:")
        for month, total_btc in btc_by_month.items():
            if month == "2024-08":
                #Subtract the bitcoins that were bought in August 2024 and September 2023.
                total_btc-=1145
                btc_by_month[month]-=1145
            elif month == "2023-09":
                total_btc-=375*5+200
                btc_by_month[month]-=375*5+200
            #print(f"{month}: {total_btc:.8f}")
        return btc_by_month
    def plot_btc_histogram(self):
        btc_by_month = self.calculate_btc_mined_per_month(self.bitcoin_data)
        # Sort the dictionary by month
        sorted_months = sorted(btc_by_month.keys())
        btc_values = [btc_by_month[month] for month in sorted_months]
        # Create the histogram
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_months, btc_values, width=0.6, align='center', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("(Year-Month)")
        plt.ylabel("Bitcoin Mined")
        plt.title("Bitcoin Mined Per Month")
        plt.tight_layout()
        plt.show()

# Only execute when this script is run directly
if __name__ == "__main__":
    stock_name = "CLSK"
    scraper = web_scraper(stock_name)
    # scraper.scrape_earnings()
    #scraper.scrape_bitcoin_address()
    fear_greed_value = scraper.scrape_fear_greed_index(scraper.sentiment_url)
    print(f"Fear & Greed Index: {fear_greed_value}")
    #scraper.scrape_bitcoin_address_all_time()
    #print(scraper.calculate_total_btc(scraper.bitcoin_data_2024))
    # print(scraper.calculate_btc_mined_per_month(scraper.bitcoin_data))
    # print(scraper.plot_btc_histogram()
