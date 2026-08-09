from fredapi import Fred
import pandas as pd
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np


# Load .env file
load_dotenv("passcodes.env")
fred = Fred(api_key=os.getenv("FRED_KEY"))


TICKER = 'SPY'
INTERVAL = '1d'
PERIOD = '730d' if INTERVAL == '1h' else 'max'

LOOKBACK = 10000


def get_data(ticker=TICKER, lookback=LOOKBACK, interval=INTERVAL):
    df = yf.download(
        ticker,
        interval=interval,
        auto_adjust=False,
        period=PERIOD,
        group_by='column'
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['Asset_Returns'] = (1 + df['Close'].pct_change()).cumprod() - 1

    subset = df.iloc[-lookback:, :]

    plt.figure(figsize=(12, 6))
    plt.plot(subset['Close'])
    plt.title(f'Price Movements for {ticker} During Study')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.show()

    return subset.dropna()


def add_inflation(df):
    inflation = pd.DataFrame(
        fred.get_series('CPIAUCSL', units='pc1'),
        columns=['Inflation']
    )

    combined = pd.concat([df, inflation], axis=1).ffill()
    return combined


def add_interest_rates(df):
    fedfunds = pd.DataFrame(
        fred.get_series('DFF'),
        columns=['FedFunds']
    )

    combined = pd.concat([df, fedfunds], axis=1).ffill()
    return combined


def add_gdp(df):
    gdp = pd.DataFrame(
        fred.get_series('GDP', units='pc1'),
        columns=['GDP_Growth']
    )

    combined = pd.concat([df, gdp], axis=1).ffill()
    return combined


def main():
    df = get_data()
    df = add_interest_rates(df)
    df = add_inflation(df)
    df = add_gdp(df)

    return df


df = main()
df = df.dropna()


plt.figure(figsize=(14, 7))

plt.plot(df['FedFunds'])
plt.plot(df['Asset_Returns'])
plt.plot(df['Inflation'])
plt.plot(df['GDP_Growth'])

plt.legend([
    'FedFunds',
    'Asset_Returns',
    'Inflation',
    'GDP Growth'
])

plt.title('Yfinance Price Info, FRED Interest Rate, Inflation, and GDP Info')
plt.xlabel('Date')
plt.grid(True, alpha=0.3)

plt.show()