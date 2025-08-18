from fredapi import Fred
import pandas as pd
from dotenv import load_dotenv
import os

# Load your .env file
load_dotenv("passcodes.env")

fred = Fred(api_key=os.getenv("FRED_KEY"))

# Get the 10-Year minus 2-Year Treasury Constant Maturity series
spread = fred.get_series("T10Y2Y")

# Convert to DataFrame for easier merging
spread = spread.to_frame(name="T10Y2Y")
spread.index = pd.to_datetime(spread.index)
print(spread.tail())