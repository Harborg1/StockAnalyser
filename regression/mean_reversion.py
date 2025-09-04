import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

TICKER = 'TSLA'
INTERVAL = '1d'
PERIOD = '730d' if INTERVAL == '1h' else 'max'

LOOKBACK = 10000
def get_data(ticker=TICKER, lookback=LOOKBACK, interval=INTERVAL):
    df = yf.download(ticker, interval=interval, auto_adjust=True, period=PERIOD)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()  # This creates a 'Date' column

    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[f'{c}_Change'] = df[c].pct_change() * 100

    subset = df.iloc[-lookback:, :]
    return subset.dropna()

def main():
    df = get_data()
    return df

def add_gap_big_moves(df: pd.DataFrame, z: float = 2.0) -> pd.DataFrame:
    """
    Adds columns for gap returns and ±zσ big-gap flags using Open_t vs Close_{t-1}.
    Big_Gap: 1 (gap up), -1 (gap down), 0 otherwise
    """
    d = df.copy()
    d['Prev_Close'] = d['Close'].shift(1)
    d['Gap_%'] = (d['Open'] / d['Prev_Close'] - 1.0) * 100.0

    mu = d['Gap_%'].mean(skipna=True)
    sd = d['Gap_%'].std(skipna=True)

    d['Big_Gap'] = np.where(
        d['Gap_%'] > mu + z * sd, 1,
        np.where(d['Gap_%'] < mu - z * sd, -1, 0)
    )
    return d

def gap_win_rates(df: pd.DataFrame, z: float = 2.0, horizon: int = 1) -> dict:
    """
    Win definition:
      - horizon=0 (default): same-day follow-through => Close_t > Open_t
      - horizon>0: follow-through from today's open to Close_{t+h} => Close_{t+h} > Open_t
    Returns counts, win rates, and avg returns (%) for big gap up and big gap down days.
    """

    d = add_gap_big_moves(df, z=z).copy()

    if horizon == 0:
        d['Future_Close'] = d['Close']  # same day
    else:
        #Future day
        d['Future_Close'] = d['Close'].shift(-horizon)

    # Return measured from today's Open to the chosen future Close
    d['Ret_%'] = (d['Future_Close'] / d['Open'] - 1.0) * 100.0

    # Need yesterday's close for the gap AND a valid future close
    valid = d.dropna(subset=['Prev_Close', 'Future_Close'])

    def summarize(mask: pd.Series) -> dict:
        subset = valid.loc[mask]
        if subset.empty:
            return {'count': 0, 'win_rate': np.nan, 'avg_ret_%': np.nan}

        wins = (subset['Future_Close'] > subset['Open']).mean()
        avg_ret = subset['Ret_%'].mean()
        return {'count': int(len(subset)), 'win_rate': float(wins), 'avg_ret_%': float(avg_ret)}
    
    res_up   = summarize(valid['Big_Gap'] == 1)   # after big gap UP days
    res_down = summarize(valid['Big_Gap'] == -1)  # after big gap DOWN days
    base     = summarize(valid['Big_Gap'].isin([1, 0, -1]))  # all valid days

    return {
        'params': {'z_sigma': z, 'horizon_days': horizon},
        'big_gap_up': res_up,
        'big_gap_down': res_down,
        'baseline_all_days': base
    }


def get_gap_trade_details(df: pd.DataFrame, z: float = 2.0, horizon: int = 1) -> pd.DataFrame:
    d = add_gap_big_moves(df, z=z).copy()

    d['Signal_Date'] = pd.to_datetime(df['Date'])

    # Required fields
    d['Prev_Close'] = d['Close'].shift(1)
    d['Prev_Close_Date'] = d['Signal_Date'].shift(1)
    d['Entry_Open'] = d['Open']
    d['Exit_Close'] = d['Close'].shift(-horizon)
    d['Exit_Date'] = d['Signal_Date'].shift(-horizon)

    # Returns
    d['Gap_%'] = (d['Entry_Open'] / d['Prev_Close'] - 1.0) * 100.0
    d['Trade_%'] = (d['Exit_Close'] / d['Entry_Open'] - 1.0) * 100.0

    d['Win'] = (d['Exit_Close'] > d['Entry_Open']).astype(int)


    # Filter to only valid rows
    d = d.dropna(subset=['Prev_Close', 'Entry_Open', 'Exit_Close'])

    # Optional: filter to just big gap up/down
    d = d[d['Big_Gap'] != 0]

    # Select useful columns
    result = d[[
        'Signal_Date',
        'Prev_Close_Date',
        'Prev_Close',
        'Entry_Open',
        'Exit_Date',
        'Exit_Close',
        'Big_Gap',
        'Gap_%',
        'Trade_%',
        'Win'
    ]].copy()

    result = result.sort_values('Signal_Date').reset_index(drop=True)
    result.attrs['params'] = {'z_sigma': z, 'horizon_days': horizon}
    return result

df = main()  # your existing pipeline

# trades = get_gap_trade_details(df, z=2.0, horizon=3)  # 0 = same 

# print(trades.tail(50))

# for i in range(0,4):
#     stats_i_days = gap_win_rates(df, z=2.0, horizon=i)
#     print("Day",i)
#     print("Win rate for big gap up", stats_i_days["big_gap_up"]["win_rate"])
#     print("Win rate for big gap down", stats_i_days["big_gap_down"]["win_rate"])
#     print("Win rate for baseline", stats_i_days["baseline_all_days"]["win_rate"])


def plot_gap_win_rates_vs_baseline(df, z=2.0, max_horizon=10):
    points = []

    for h in range(max_horizon + 1):
        stats = gap_win_rates(df, z=z, horizon=h)
        x = stats['big_gap_up']['win_rate']
        y = stats['big_gap_down']['win_rate']
        baseline = stats['baseline_all_days']['win_rate']
        points.append((x, y, h, baseline))

    # Extract x, y, and horizon for plotting
    xs = [x for x, _, _, _ in points]
    ys = [y for _, y, _, _ in points]
    labels = [h for _, _, h, _ in points]
    baseline = points[0][3]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, color='blue')
    plt.scatter(baseline,baseline, color ="red")
    plt.text(baseline, baseline, f'Baseline win rate', fontsize=8, ha='left', va='bottom')

    for i in range(len(xs)):
        plt.text(xs[i], ys[i], f'{labels[i]}d', fontsize=8, ha='right', va='bottom')

    plt.xlabel('Big Gap Up Win Rate')
    plt.ylabel('Big Gap Down Win Rate')
    plt.title(f'Gap Strategy Win Rates (z={z})\nBaseline = {baseline:.2%}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Call this after df = main()
plot_gap_win_rates_vs_baseline(df, z=2.0, max_horizon=10)
