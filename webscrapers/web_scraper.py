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
import platform
from webdriver_manager.chrome import ChromeDriverManager

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
        self.driver = None
        self.bitcoin_data = "json_folder\\bitcoin_address_data_all_time.json"
        self.bitcoin_data_2024 = "json_folder\\bitcoin_address_data_2024.json"
        self.jobs_release ="json_folder\\jobs_release_dates.json"

    def setup_driver(self):
            options = Options()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            # Tip: Det er ofte bedre at fjerne fastlåst user-agent, hvis du vil undgå detektering
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

            # Headless mode hvis det ikke er Windows
            if platform.system() != "Windows":
                options.add_argument("--headless=chrome")

            # KORREKT MÅDE: Brug ChromeDriverManager til at finde stien, og send options med
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)

            self.driver.get("https://www.google.dk")
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
    
    def scrape_jobs_release_dates(self, output_path):
        """Scrape Employment Situation release dates from the US bureau of labor statistics.
        Saves them to the JSON file at output_path."""

        # Ensure folder exists
        json_folder = os.path.dirname(output_path) or "."
        os.makedirs(json_folder, exist_ok=True)

        url = "https://www.bls.gov/schedule/news_release/empsit.htm"
        driver = self.setup_driver()
        driver.get(url)

        try:
            # Wait for the main content table to load
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.ID, "main-content-td"))
            )
            soup = BeautifulSoup(driver.page_source, "html.parser")
        except Exception as e:
            print(f"Error loading Jobs Release page: {e}")
            driver.quit()
            return []
        finally:
            driver.quit()

        # Narrow down to the main-content-td
        content_td = soup.find("td", {"id": "main-content-td"})
        if not content_td:
            print("Could not find main-content-td")
            return []

        # Now find the release schedule table inside
        table = content_td.find("table")
        if not table:
            print("Could not find release schedule table")
            return []

        rows = table.find_all("tr")[1:]  # skip header
        data = []
        for r in rows:
            cols = [c.get_text(strip=True) for c in r.find_all("td")]
            if len(cols) >= 3:
                reference_month, release_date, release_time = cols[:3]

                # Parse release_date like "Jan. 10, 2025"
                try:
                    dt = datetime.strptime(release_date, "%b. %d, %Y")
                    release_date_iso = dt.strftime("%Y-%m-%d")
                except ValueError:
                    # fallback if it doesn't match
                    release_date_iso = release_date  

                data.append({
                    "reference_month": reference_month,
                    "release_date": release_date_iso,
                    "release_time": release_time
                })

        # Save to JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"Saved {len(data)} records to {output_path}")
        return data


        
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

        url = "https://investors.cleanspark.com/news/"
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
            soup = BeautifulSoup(driver.page_source, "html.parser")
        finally:
            pass

        articles = soup.select("a[href*='/news/news-details/']")
        new_links = []

        for article in articles:
            title = article.get_text(strip=True)
            link = article["href"]
            if not link.startswith("http"):
                link = "https://investors.cleanspark.com" + link

            if any(item["link"] == link for item in existing_links):
                continue

            driver.get(link)
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "_ctrl0_ctl55_divNewsDetailsDateTime"))
                )
                article_soup = BeautifulSoup(driver.page_source, "html.parser")
                date_element = article_soup.find("div", id="_ctrl0_ctl55_divNewsDetailsDateTime")
                article_date = date_element.get_text(strip=True) if date_element else None
            except Exception as e:
                print(f"Error retrieving date for {link}: {e}")
                article_date = None  # 👈 important change here
            new_links.append(
                {
                    "title": title,
                    "link": link,
                    "date": article_date,
                }
            )

        driver.quit()

        if new_links:
            with open(json_file_path, "w", encoding="utf-8") as file:
                json.dump(existing_links + new_links, file, indent=4, ensure_ascii=False)
            print(f"Added {len(new_links)} new article(s)")
        else:
            print("No recent news")

        return new_links



    def scrape_fear_greed_index(self, url):
        if self.driver is None:
            self.setup_driver()
        self.driver.get(url)

        json_file_path = os.path.join("json_folder", "feargreed.json")

        try:
            time.sleep(3)
            locator = (By.CLASS_NAME, "market-fng-gauge__dial-number-value")
            WebDriverWait(self.driver, 15).until(EC.presence_of_element_located(locator))
            elements = self.driver.find_elements(By.CLASS_NAME, "market-fng-gauge__dial-number-value")

            for el in elements:
                value = el.text.strip()
                if value:
                    new_entry = {
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "fear_greed_index": value
                    }

                    # Load existing data if file exists, otherwise start a list
                    if os.path.exists(json_file_path):
                        with open(json_file_path, "r") as f:
                            try:
                                data = json.load(f)
                                if not isinstance(data, list):
                                    data = [data]
                            except json.JSONDecodeError:
                                data = []
                    else:
                        data = []
                    data.append(new_entry)

                    # Save updated data
                    with open(json_file_path, "w") as f:
                        json.dump(data, f, indent=4)

                    self.driver.quit()
                    return value

            return None
        except Exception as e:
            print(f"Error scraping Fear & Greed index: {e}")
            return None

    

    def scrape_bitcoin_address(self):
        """
        Scrapes data from the bitcoin address of CLSK that is more recent than the cutoff date."""
        # Calculate target_count as the number of days from today to May 1st
        self.setup_driver()
        today = datetime.now()
        cutoff_date = datetime(2025, 5, 1)
        days_difference = (today - cutoff_date).days+1
        target_count = days_difference
        print(f"Target count (days difference): {target_count}")
        url = "https://bitref.com/3KmNWUNVGoTzHN8Cyc1kVhR1TSeS6mK9ab"
        json_file_path = self.bitcoin_data_2024
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
        bitcoin_holding = 12090 #Estimated bitcoin holding since 01-05-2025
        off_set=900 #A transaction of 900 BTC that occured on 15-05-2025 (not mined)
        bitcoin_holding-=off_set 
        bitcoin_data = []
        if os.path.exists(json_data):
            with open(json_data, "r", encoding="utf-8") as file:
                bitcoin_data = json.load(file)

        total_sum = bitcoin_holding

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

    def scrape_coinglass_change(self, attempts=5):
        """Scrapes the 24h Change value from Coinglass Balance page and saves to JSON."""
        url = "https://www.coinglass.com/Balance"
        json_path = os.path.join("json_folder", "coinglass_balance_24h_change.json")

        for attempt in range(1, attempts + 1):
            try:
                print(f"🔁 Attempt {attempt}/{attempts}")
                self.driver.get(url)
                try:
                    consent_btn = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[normalize-space()='Consent']"))
                    )
                    consent_btn.click()
                    print("Clicked Consent button")
                    time.sleep(1)
                
                except Exception:
                         print("No consent popup found")

                # Get 24h percentage change             
                value_change = WebDriverWait(self.driver, 10).until(
                            EC.presence_of_all_elements_located((
                                By.XPATH,
                                "//div[contains(@class, 'Number undefined') and (contains(@class, 'fall-color') or contains(@class, 'rise-color'))]"
                            ))
                        )

                pct_chg = None
                for j, el in enumerate(value_change):
                    text = el.text.strip()
                    if text and j == 0:
                        pct_chg = text
                        break
                if not pct_chg:
                    raise ValueError("Failed to find 24h % change.")

                # Scroll to reveal table
                self.driver.execute_script("window.scrollBy(0, 1400);")
                scroll_container = WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "ant-table-body"))
                )
                self.driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_container)
                time.sleep(1.5)
                
                # Try to find the total row specifically
                try:
                    total_row = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, "//tr[td[contains(., 'Total')]]"))
                    )

                    # Find the right-aligned numeric cell (the total BTC value)
                    val_btc = total_row.find_element(
                        By.XPATH, ".//td[@class='ant-table-cell' and @style='text-align: right;'][1]/div"
                    ).text.strip()

                    print("Total BTC value:", val_btc)

                except Exception as e:
                    print(" Could not find total BTC row:", e)

                # Prepare data
                data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Volume Percentage Change (24h)": pct_chg,
                    "Total bitcoin": val_btc,
                }

                # Save to JSON
                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                else:
                    existing = []

                existing.append(data)

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=4)

                print(f"Saved Coinglass data: {data}")
                return  # Success → exit function

            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt == attempts:
                    print("All attempts failed")


    def scrape_useful_data(self):
        from datetime import datetime
        def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
        self.driver = self.setup_driver()
        try:
            log("🔍 Starting Coinglass scrape...")
            self.scrape_coinglass_change()
            log("✅ Finished Coinglass.")

        finally:
            self.driver.quit()

# Only execute when this script is run directly
if __name__ == "__main__":
    stock_name = "CLSK"
    scraper = web_scraper(stock_name)
    #scraper.scrape_articles()
    scraper.scrape_useful_data()
    #scraper.scrape_jobs_release_dates(scraper.jobs_release)
    # scraper.scrape_earnings()
    #scraper.scrape_bitcoin_address()
    #scraper.scrape_coinglass_change()
    # print(f"Fear & Greed Index: {fear_greed_value}")
    #scraper.scrape_bitcoin_address_all_time()
    #print(scraper.calculate_total_btc(scraper.bitcoin_data_2024))
    # print(scraper.calculate_btc_mined_per_month(scraper.bitcoin_data))
    # print(scraper.plot_btc_histogram()
