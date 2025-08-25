from fredapi import Fred
import pandas as pd
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import yfinance as yf

"""
A yield curve inversion is one of the strongest predictors of economic recessions.
This tool visualizes recent financial crises and shows how long it took for the market to bottom after the inversion.
"""

# Load .env file
load_dotenv("passcodes.env")
fred = Fred(api_key=os.getenv("FRED_KEY"))

# Define both crisis events
crisis_data = [
    {
        "label": "1980 Bear Market (Volcker #1)",
        "range": ("1978-01-01", "1981-12-31"),
        "inversion": "1978-08-18",   # approx, should let get_spread confirm
        "bottom": "1980-03-27",      # S&P 500 bottom
        "color": "red"
    },
    {
        "label": "1981–1982 Bear Market (Volcker #2)",
        "range": ("1980-01-01", "1983-12-31"),
        "inversion": "1980-01-02",   # approx, should let get_spread confirm
        "bottom": "1982-08-12",      # famous 1982 low
        "color": "teal"
    },

    {
        "label": "1990 Bear Market (S&L + Gulf War)",
        "range": ("1988-01-01", "1991-12-31"),
        "inversion": "1988-12-13",   # this matches what your method found
        "bottom": "1990-10-11",      # October 1990 bottom
        "color": "brown"
    },


    {
        "label": "Dotcom Bubble",
        "range": ("1999-01-01", "2003-12-31"),
        "inversion": "2000-02-02",
        "bottom": "2002-10-09",
        "color": "purple"
    },
    {
        "label": "Global Financial Crisis",
        "range": ("2006-01-01", "2010-12-31"),
        "inversion": "2006-01-31",
        "bottom": "2009-03-09",
        "color": "darkgreen"
    },
    {
        "label": "COVID-19 Crash",
        "range": ("2019-01-01", "2021-12-31"),
        "inversion": "2019-08-27",
        "bottom": "2020-03-23",
        "color": "blue"
    },
    {
        "label": "2022-2023 Bear Market",
        "range": ("2022-01-01", "2023-12-31"),  
        "inversion": "2022-04-01",        
        "bottom": "2022-10-13",                 
        "color": "firebrick"
    }
]

# Function to get and prepare spread data for a given date range
def get_spread(start, end, plot=False):
    spread = fred.get_series("T10Y2Y").to_frame(name="T10Y2Y")
    spread = spread.loc[start:end]
    spread = spread.resample("D").interpolate(method="linear")
    spread.index = pd.to_datetime(spread.index)

    # Find first inversion date (first time spread < 0)
    inversion_dates = spread[spread["T10Y2Y"] < 0].index
    first_inversion = inversion_dates.min() if not inversion_dates.empty else None

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(spread.index, spread["T10Y2Y"], label="10Y - 2Y Spread", color="black")
        plt.axhline(0, color="red", linestyle="--", linewidth=1, label="Inversion Line (0%)")
        plt.xlabel("Date")
        plt.ylabel("Spread (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return spread, first_inversion

def calculate_spy_drop(crisis_data, output_path="spy_crisis_drops.csv"):
    results = []

    for crisis in crisis_data:
        _, end = crisis["range"]
        start = crisis["inversion"]
        bottom_date = pd.to_datetime(crisis["bottom"])

        ticker = "^GSPC" if pd.to_datetime(start).year < 1993 else "SPY"

        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        df.index = pd.to_datetime(df.index)

        if df.empty or bottom_date not in df.index:
            print(f"Missing SPY data for {crisis['label']} at {bottom_date.date()}. Skipping.")
            continue

        close_prices = df["Close"][ticker]
        start_price = close_prices.iloc[0]
        bottom_price = close_prices.loc[bottom_date]

        pct_change = ((bottom_price - start_price) / start_price) * 100

        results.append({
            "label": crisis["label"],
            "start_date": df.index[0].date(),
            "bottom_date": bottom_date.date(),
            "start_price": round(start_price, 2),
            "bottom_price": round(bottom_price, 2),
            "pct_change": round(pct_change, 2)
        })

    df_result = pd.DataFrame(results)
    df_result.to_csv(output_path, index=False)
    print(f"\n✅ Clean CSV saved to: {output_path}")

    return df_result

n = len(crisis_data)
cols = 2
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(10, 2 * rows), sharey=False)

axes = axes.flatten()

for ax, crisis in zip(axes, crisis_data):
    spread, first_inversion = get_spread(*crisis["range"])

    inversion_date = first_inversion
    bottom_date = pd.to_datetime(crisis["bottom"])
    days_between = (bottom_date - inversion_date).days

    if inversion_date < spread.index[0] or bottom_date < spread.index[0]:
        raise ValueError("Inversion or bottom date is outside the data range.")
    
    nearest_bottom = spread.index[spread.index.get_indexer([bottom_date], method="nearest")[0]]
    bottom_value = spread.loc[nearest_bottom, "T10Y2Y"]

    ax.plot(spread.index, spread["T10Y2Y"], label="10Y - 2Y Spread", color="black")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.axvline(first_inversion, color="orange", linestyle="--", label=f"Inversion ({inversion_date.date()})")
    ax.axvline(nearest_bottom, color=crisis["color"], linestyle="--", label=f"Market Bottom ({bottom_date.date()})")

    ax.set_title(f"{crisis['label']} ({crisis['range'][0][:4]}–{crisis['range'][1][:4]})")
    ax.set_ylabel("Spread (%)")
    ax.grid(True)
    ax.legend(fontsize=5.5, frameon=True, framealpha=0.8, fancybox=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

# Remove unused subplot if only 3 crises
if len(crisis_data) < len(axes):
    for j in range(len(crisis_data), len(axes)):
        fig.delaxes(axes[j])

plt.subplots_adjust(wspace=0.8, hspace=0.8)
plt.tight_layout()
plt.show()

df_results = calculate_spy_drop(crisis_data, output_path="csv_files/spy_drops.csv")
print(df_results)

spread = fred.get_series("T10Y2Y").to_frame(name="T10Y2Y")

start = spread.index.min()

end = spread.index.max()

get_spread(start=start,end=end, plot=True)
