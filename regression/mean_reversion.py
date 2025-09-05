import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

output_path = "csv_files/mean_reversion_results.csv"
BASE = "SPY"
TICKER = 'TSLA'
INTERVAL = '1d'
PERIOD = '730d' if INTERVAL == '1h' else 'max'
LOOKBACK = 100

def get_data(ticker:str, lookback=LOOKBACK, interval=INTERVAL):
    df = yf.download(ticker, interval=interval, auto_adjust=True, period=PERIOD)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()  # This creates a 'Date' column


    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[f'{c}_Change'] = df[c].pct_change() * 100

    subset = df.iloc[-lookback:, :]
    return subset.dropna()

def add_big_gap_moves(df: pd.DataFrame, z: float) -> pd.DataFrame:
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

def calculate_subset_metrics(valid: pd.DataFrame, mask: pd.Series) -> dict:
    subset = valid.loc[mask]
    if subset.empty:
        return {'count': 0, 'win_rate': np.nan, 'avg_ret_%': np.nan, 'p&L': np.nan}

    wins = (subset['Future_Close'] > subset['Open']).mean()
    avg_ret = subset['Ret_%'].mean()
    profit = subset['p&L'].sum()
    return {
        'count': int(len(subset)),
        'win_rate': float(wins),
        'avg_ret_%': float(avg_ret),
        'p&L': float(profit)
    }

def gap_win_rates(df: pd.DataFrame, z: float, horizon: int) -> dict:
    """
    Win definition:
      - horizon=0: same-day follow-through => Close_t > Open_t
      - horizon>0: follow-through from today's open to Close_{t+h} => Close_{t+h} > Open_t
    Returns counts, win rates, and avg returns (%) for big gap up and big gap down days.
    """

    d = add_big_gap_moves(df, z=2).copy()

    if horizon == 0:
        d['Future_Close'] = d['Close']
    else:
        d['Future_Close'] = d['Close'].shift(-horizon)

    d['Ret_%'] = (d['Future_Close'] / d['Open'] - 1.0) * 100.0
    d['p&L'] = d['Future_Close'] - d['Open']

    valid = d.dropna(subset=['Prev_Close', 'Future_Close'])
    res_up = calculate_subset_metrics(valid, valid['Big_Gap'] == 1)
    res_down = calculate_subset_metrics(valid, valid['Big_Gap'] == -1)
    base = calculate_subset_metrics(valid, valid['Big_Gap'].isin([1, 0, -1]))

    return {
        'params': {'z_sigma': z, 'horizon_days': horizon},
        'big_gap_up': res_up,
        'big_gap_down': res_down,
        'baseline_all_days': base
    }

def get_gap_trade_details(df: pd.DataFrame, z: float, horizon: int) -> pd.DataFrame:
    drop_overlapping = True
    d = add_big_gap_moves(df, z=z).copy()
    d['Signal_Date'] = pd.to_datetime(d['Date'])
    d = d.sort_values('Signal_Date')

    # Baseline trade fields
    d['Prev_Close'] = d['Close'].shift(1)
    d['Entry_Open'] = d['Open']
    d['Exit_Close'] = d['Close'].shift(-horizon)
    d['Exit_Date'] = d['Signal_Date'].shift(-horizon)
    d['Trade_%'] = (d['Exit_Close'] / d['Entry_Open'] - 1.0) * 100
    d['Profit_and_loss_pct'] = (d['Exit_Close'] - d['Entry_Open']) / d['Entry_Open'] * 100

    # SPY baseline aligned by dates
    spy = get_data(BASE)[['Date', 'Open', 'Close']].copy()
    spy['Date'] = pd.to_datetime(spy['Date'])
    spy = spy.sort_values('Date')
    spy = spy.rename(columns={'Open': 'Open_SPY', 'Close': 'Close_SPY'})

    # SPY open at ticker signal date
    merged = d.merge(
        spy[['Date', 'Open_SPY']],
        left_on='Signal_Date',
        right_on='Date',
        how='left'
    ).rename(columns={'Open_SPY': 'Open_SPY_at_signal'})

    # SPY close at ticker exit date
    merged = merged.merge(
        spy[['Date', 'Close_SPY']],
        left_on='Exit_Date',
        right_on='Date',
        how='left'
    ).rename(columns={'Close_SPY': 'Close_SPY_at_exit'})

    merged = merged.drop(columns=['Date'])

    # Baseline SPY % return over same window
    merged['Profit_and_loss_baseline_pct'] = (
        (merged['Close_SPY_at_exit'] - merged['Open_SPY_at_signal'])
        / merged['Open_SPY_at_signal'] * 100
    )

    # Keep only valid big-gap rows and fully aligned windows
    merged = merged[merged['Big_Gap'] != 0]
    merged = merged.dropna(subset=['Exit_Close', 'Open_SPY_at_signal', 'Close_SPY_at_exit'])

    merged = merged.sort_values('Signal_Date').reset_index(drop=True)
    merged.attrs['params'] = {'z_sigma': z, 'horizon_days': horizon}
    merged = merged[[
        'Signal_Date', 'Exit_Date',
        "Prev_Close",
        'Entry_Open', 'Exit_Close',
        'Big_Gap', 'Profit_and_loss_pct',
        'Open_SPY_at_signal', 'Close_SPY_at_exit', 'Profit_and_loss_baseline_pct'
    ]]
    merged.to_csv(output_path)

    # optional: drop overlapping trades before compounding
    t = merged.copy()
    if drop_overlapping:
        keep = []
        last_exit = pd.Timestamp.min
        for _, row in t.iterrows():
            if row['Signal_Date'] >= last_exit:
                keep.append(True)
                last_exit = row['Exit_Date']
            else:
                keep.append(False)
        t = t.loc[keep].reset_index(drop=True)

    # compounded totals
    base_mult = (1 + t['Profit_and_loss_pct'] / 100.0).prod()
    spy_mult  = (1 + t['Profit_and_loss_baseline_pct'] / 100.0).prod()

    stats = {
        'n_trades': int(len(t)),
        f'{TICKER}_total_%': round(float((base_mult - 1) * 100.0),2),
        'spy_total_%': round(float((spy_mult - 1) * 100.0),2),
        'relative_edge_%': round(float(((base_mult / spy_mult) - 1) * 100.0),2),
        f'{TICKER}_avg_%': round(float(t['Profit_and_loss_pct'].mean()),2),
        'spy_avg_%': round(float(t['Profit_and_loss_baseline_pct'].mean()),2),
        f'{TICKER}_beats_spy_rate_%': round(float((t['Profit_and_loss_pct'] > t['Profit_and_loss_baseline_pct']).mean() * 100.0),2)
    }

    return merged,stats


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

df = get_data(TICKER)
trades,stats = get_gap_trade_details(df, z=2.0, horizon=3) 

print(stats)


# stats_3_days = gap_win_rates(df, z=2.0, horizon=3)

# for i in range(0,4):
#     stats_i_days = gap_win_rates(df, z=2.0, horizon=i)
#     print("Day",i)
#     print("Win rate for big gap up", stats_i_days["big_gap_up"]["win_rate"])
#     print("Win rate for big gap down", stats_i_days["big_gap_down"]["win_rate"])
#     print("Win rate for baseline", stats_i_days["baseline_all_days"]["win_rate"])
#     print("P&L for big gap up", stats_i_days["big_gap_up"]["p&L"])
#     print("P&L for big gap down", stats_i_days["big_gap_down"]["p&L"])


# plot_gap_win_rates_vs_baseline(df, z=2.0, max_horizon=10)
