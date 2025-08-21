from fredapi import Fred
import pandas as pd
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


"""
A yield curve inversion is one of the strongest predictors of economic recessions.
This tool visualizes recent financial crises and shows how long it took for the market to bottom after the inversion.
"""


# Load .env file
load_dotenv("passcodes.env")
fred = Fred(api_key=os.getenv("FRED_KEY"))

# Function to get and prepare spread data for a given date range
def get_spread(start, end):
    spread = fred.get_series("T10Y2Y").to_frame(name="T10Y2Y")
    spread = spread.loc[start:end]
    spread = spread.resample("D").interpolate(method="linear")
    spread.index = pd.to_datetime(spread.index)
    return spread


# Define both crisis events
crisis_data = [
    {
        "label": "Global Financial Crisis",
        "range": ("2006-01-01", "2010-12-31"),  # expanded range to include inversion
        "inversion": "2006-07-01",
        "bottom": "2009-03-09",
        "color": "darkgreen"
    },
    {
        "label": "COVID-19 Crash",
        "range": ("2019-01-01", "2021-12-31"),
        "inversion": "2019-08-01",
        "bottom": "2020-03-23",
        "color": "blue"
    }
]

# Plot with separate y-axes
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

for ax, crisis in zip(axes, crisis_data):
    spread = get_spread(*crisis["range"])

    inversion_date = pd.to_datetime(crisis["inversion"])
    bottom_date = pd.to_datetime(crisis["bottom"])
    days_between = (bottom_date - inversion_date).days

    # Get nearest available dates
    if inversion_date < spread.index[0] or bottom_date < spread.index[0]:
        raise ValueError("Inversion or bottom date is outside the data range.")

    nearest_inversion = spread.index[spread.index.get_indexer([inversion_date], method="nearest")[0]]
    nearest_bottom = spread.index[spread.index.get_indexer([bottom_date], method="nearest")[0]]
    bottom_value = spread.loc[nearest_bottom, "T10Y2Y"]

    # Plot spread
    ax.plot(spread.index, spread["T10Y2Y"], label="10Y - 2Y Spread", color="black")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)

    # Mark inversion and bottom
    ax.axvline(nearest_inversion, color="orange", linestyle="--", label=f"Inversion ({inversion_date.date()})")
    ax.axvline(nearest_bottom, color=crisis["color"], linestyle="--", label=f"Market Bottom ({bottom_date.date()})")


    ax.set_title(f"{crisis['label']} ({crisis['range'][0][:4]}–{crisis['range'][1][:4]})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Spread (%)")
    ax.grid(True)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

# Adjust spacing between plots
plt.subplots_adjust(wspace=0.6)  # wider space between plots
plt.tight_layout()
plt.show()


