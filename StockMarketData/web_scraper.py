# earnings_scraper.py

import json
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import datetime

#https://electrek.co/?s=tesla

class web_scraper:
    def __init__(self, stock_name):
        """Initialize with the stock name (ticker)."""
        self.stock_name = stock_name.upper()
        self.earnings_url = f'https://www.nasdaq.com/market-activity/stocks/{self.stock_name}/earnings'
        self.cpi_url = "https://www.bls.gov/schedule/news_release/cpi.htm"
        self.json_file_path_earnings = f"json_folder\\stock_earnings.json"
        self.json_file_path_cpi = f"json_folder\\cpi.json"
        self.driver = None
        self.bitcoin_data = "json_folder\\bitcoin_address_data.json"

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


    def scrape_earnings(self):
        """Scrapes earnings dates and saves them to a JSON file."""
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

        # Select earnings dates elements
        earnings_elements = soup.select(".announcement-date")
        new_earnings = []

        # Extract each earnings item
        for element in earnings_elements:
            date_text = element.get_text(strip=True)
            
            # Skip if this date is already in existing_earnings
            if any(item['date'] == date_text for item in existing_earnings):
                continue
            

            # Append new earnings data
            new_earnings.append({
                "date": date_text,
                "stock": self.stock_name
            })

        self.driver.quit()

        # Save new earnings data if any found
        if new_earnings:
            with open(self.json_file_path_earnings, "w", encoding="utf-8") as file:
                json.dump(existing_earnings + new_earnings, file, indent=4, ensure_ascii=False)
            print(f"Added {len(new_earnings)} new earnings date(s) for {self.stock_name}")
        else:
            print("No new earnings dates found")

        return new_earnings
    
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
        Scrapes data from the specified Bitcoin address page that is the bitcoin address of CLSK.
        The target count is the number of days between today and December 4th 2024.
        """
        # Calculate target_count as the number of days from today to December 4th
        today = datetime.now()
        cutoff_date = datetime(2024, 12, 4)

        days_difference = (today - cutoff_date).days
        if days_difference <= 0:
            print("December 4th has not passed yet. No data to scrape.")
            return []
        target_count = days_difference
        print(f"Target count (days difference): {target_count}")

        url = "https://bitref.com/3KmNWUNVGoTzHN8Cyc1kVhR1TSeS6mK9ab"
        json_file_path = self.bitcoin_data
        self.setup_driver()
        self.driver.get(url)

        # Load existing data if available
        existing_data = []
        if os.path.exists(json_file_path):
            with open(json_file_path, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
                
        total_clicks = 0
        while len(existing_data) < target_count:
            # Otherwise, click "Load More Transactions" to load more data
            try:
                load_more_button = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='button'][class*='btn-outline-secondary'][onclick*='getTransactions']"))
                )
                self.driver.execute_script("arguments[0].click();", load_more_button)
                WebDriverWait(self.driver, 10).until(
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

            elements = table_body.find_all("td", class_="text-end text-success")
            new_data = []

            for element in elements:
                data_text = element.get_text(strip=True)

                # Skip if the data is already in the existing data
                if any(item['btc_mined'] == data_text for item in existing_data):
                    continue

                # Append new data
                new_data.append({"btc_mined": data_text})

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

    def calculate_total_btc(self,json_data):
         # Load Bitcoin data from the file
        dec_3_2024 = 20.97159456
        dec_2_2024 = 21.31648126
        dec_1_2024 = 20.94454838
        nov_30_2024 = 9297
        total_btc_as_of_december_4th_2024 = nov_30_2024+dec_1_2024+dec_2_2024+dec_3_2024
        # Source: https://investors.cleanspark.com/news/news-details/2024/CleanSpark-Releases-November-2024-Bitcoin-Mining-Update/default.aspx
        bitcoin_data = []
        if os.path.exists(json_data):
            with open(json_data, "r", encoding="utf-8") as file:
                bitcoin_data = json.load(file)

        total_sum = total_btc_as_of_december_4th_2024

        for item in bitcoin_data:
            # Convert the 'data' string to a float and add it to the total
            try:
                value = float(item['btc_mined'].replace('+', ''))
                total_sum += value
            except ValueError:
                print(f"Skipping invalid value: {item['btc_mined']}")
        return total_sum

# Only execute when this script is run directly
if __name__ == "__main__":
    stock_name = "CLSK"
    scraper = web_scraper(stock_name)
    # scraper.scrape_earnings()
    scraper.scrape_bitcoin_address()
    print(scraper.calculate_total_btc(scraper.bitcoin_data))
