
"""
Tools for analyzing and backtesting a mean reversion trading strategy based on large price gaps.
Includes functions for data retrieval, signal generation, trade metrics calculation, and strategy visualization.
"""

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Get previous version of file from the 11/9/2025 if you want:)
output_path = "csv_files/mean_reversion_results.csv"
BASE = "SPY"
TICKER = 'NVO'
INTERVAL = '1d'
PERIOD = '730d' if INTERVAL == '1h' else 'max'
LOOKBACK = 365

def get_data(ticker:str, lookback=LOOKBACK, interval=INTERVAL):
    df = yf.download(ticker, interval=interval, auto_adjust=True, period=PERIOD,progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()  # This creates a 'Date' column

    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[f'{c}_Change'] = df[c].pct_change() * 100

    subset = df.iloc[-lookback:, :]
    return subset.dropna()

def add_big_gap_moves(df: pd.DataFrame, z: float) -> pd.DataFrame:
    """
    Adds columns for big gap signals based on Close_Change.
    Signal is assigned to day t, but trade entry happens on t+1.
    """
    df = df.copy()
    close_change_avg = df['Close_Change'].mean()
    close_change_std = df['Close_Change'].std()


    df['Big_Gap'] = np.where(
        df['Close_Change'] > close_change_avg + (close_change_std * z), 1,
        np.where(df['Close_Change'] < close_change_avg - (close_change_std * z), -1, 0)
    )

    return df

def calculate_subset_metrics(valid: pd.DataFrame, mask: pd.Series) -> dict:
    subset = valid.loc[mask]
    if subset.empty:
        return {'count': 0, 'win_rate': np.nan, 'avg_ret_%': np.nan, 'p&L': np.nan}
    
    # A win is futuure close > entry open
    wins = (subset['Future_Close'] > subset['Entry_Open']).mean()
    
    avg_ret = subset['Ret_%'].mean()
    profit = subset['p&L'].sum()
    return {
        'count': int(len(subset)),
        'win_rate': float(wins),
        'avg_ret_%': float(avg_ret),
        'p&L': float(profit)
    }


def gap_win_rates(df: pd.DataFrame, z: float, horizon: int) -> dict:
    d = add_big_gap_moves(df, z=z).copy()

    # shift entry forward by 1
    d['Entry_Open'] = d['Open'].shift(-1)

    if horizon == 0:
        d['Future_Close'] = d['Close'].shift(-1)
    else:
        d['Future_Close'] = d['Close'].shift(-(horizon+1))

    d['Ret_%'] = (d['Future_Close'] / d['Entry_Open'] - 1.0) * 100.0
    d['p&L'] = d['Future_Close'] - d['Entry_Open']

    valid = d.dropna(subset=['Future_Close'])
    print("Horizon:",horizon)
    res_up = calculate_subset_metrics(valid, valid['Big_Gap'] == 1)
    print("res_up", res_up["win_rate"])
    res_down = calculate_subset_metrics(valid, valid['Big_Gap'] == -1)
    print("res_down",res_down["win_rate"])
    base = calculate_subset_metrics(valid, valid['Big_Gap']==0)

    print("base", base["win_rate"])

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

    # shift entry forward by 1 day
    d['Prev_Close'] = d['Close'].shift(1).round(2)
    d['Entry_Open'] = d['Open'].shift(-1).round(2)
    d['Exit_Close'] = d['Close'].shift(-(horizon+1)).round(2)
    d['Exit_Date'] = d['Signal_Date'].shift(-(horizon+1)).round(2)

    d['Trade_%'] = ((d['Exit_Close'] / d['Entry_Open'] - 1.0) * 100).round(2)
    d['Profit_and_loss_pct'] = (((d['Exit_Close'] - d['Entry_Open']) / d['Entry_Open']) * 100).round(2)

    # add close range tuple (close[i-1], close[i])
    d['Close_Range'] = list(zip(d['Close'].shift(1).round(2), d['Close'].round(2)))

    # SPY baseline aligned by dates
    spy = get_data(BASE)[['Date', 'Open', 'Close']].copy()
    spy['Date'] = pd.to_datetime(spy['Date'])
    spy = spy.sort_values('Date')
    spy = spy.rename(columns={'Open': 'Open_SPY', 'Close': 'Close_SPY'})

    # Round to 2 decimal places
    spy['Open_SPY'] = spy['Open_SPY'].round(2)
    spy['Close_SPY'] = spy['Close_SPY'].round(2)

    merged = d.merge(
        spy[['Date', 'Open_SPY']],
        left_on='Signal_Date',
        right_on='Date',
        how='left'
    ).rename(columns={'Open_SPY': 'Open_SPY_at_signal'})

    merged = merged.merge(
        spy[['Date', 'Close_SPY']],
        left_on='Exit_Date',
        right_on='Date',
        how='left'
    ).rename(columns={'Close_SPY': 'Close_SPY_at_exit'})

    merged = merged.drop(columns=['Date'])

    merged['Profit_and_loss_baseline_pct'] =round((merged['Close_SPY_at_exit'] - merged['Open_SPY_at_signal']) / merged['Open_SPY_at_signal'] * 100,2)

    #Filter out normal days
    merged = merged[merged['Big_Gap'] != 0]
    #Drop Nans 
    merged = merged.dropna(subset=['Exit_Close', 'Open_SPY_at_signal', 'Close_SPY_at_exit'])

    merged = merged.sort_values('Signal_Date').reset_index(drop=True)
    merged.attrs['params'] = {'z_sigma': z, 'horizon_days': horizon}
    merged = merged[[
        'Signal_Date', 'Exit_Date',
        "Prev_Close", 'Close_Range',
        'Entry_Open', 'Exit_Close',
        'Big_Gap', 'Profit_and_loss_pct',
        'Open_SPY_at_signal', 'Close_SPY_at_exit', 'Profit_and_loss_baseline_pct'
    ]]


    # optional: drop overlapping trades
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

    t.to_csv(output_path)

    base_mult = (1 + t['Profit_and_loss_pct'] / 100.0).prod()
    spy_mult  = (1 + t['Profit_and_loss_baseline_pct'] / 100.0).prod()

    stats = {
        'n_trades': int(len(t)),
        f'{TICKER}_total_%': round(float((base_mult - 1) * 100.0), 2),
        'spy_total_%': round(float((spy_mult - 1) * 100.0), 2),
        'relative_edge_%': round(float(((base_mult / spy_mult) - 1) * 100.0), 2),
        f'{TICKER}_avg_%': round(float(t['Profit_and_loss_pct'].mean()), 2),
        'spy_avg_%': round(float(t['Profit_and_loss_baseline_pct'].mean()), 2),
        f'{TICKER}_beats_spy_rate_%': round(float((t['Profit_and_loss_pct'] > t['Profit_and_loss_baseline_pct']).mean() * 100.0), 2)
    }

    return t, stats



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
def generate_equity_curve(trades_df: pd.DataFrame, initial_capital: float = 10000, plot: bool = True) -> pd.Series:
    # Load SPY and TICKER data
    base = get_data(f'{BASE}')
    ticker = get_data(f'{TICKER}')

    for df in (base, ticker):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

    # Add daily returns
    base['SPY_Return_close'] = base['Close'].pct_change().fillna(0)
    base['SPY_Return_overnight'] = base['Open'] / base['Close'].shift(1) - 1
    ticker[f'{TICKER}_Return_close'] = ticker['Close'].pct_change()
    ticker[f'{TICKER}_Daily_gain'] = (ticker['Close'] / ticker['Open'] - 1).round(4)
    # Align dates
    all_days = base.index.union(ticker.index).sort_values()
    capital = initial_capital
    equity = []

    in_trade = False
    exit_date = None
    reenter_spy_next_open = False  # flag for the first day after trade exit

    for current_date in all_days:
        if current_date not in base.index or current_date not in ticker.index:
            continue  # skip non-trading days

        print(f"\nDate: {current_date.date()}, in_trade={in_trade}, capital={capital:.2f}")

        first_day_of_trade = trades_df.loc[trades_df['Signal_Date']+pd.Timedelta(days=1) ==  current_date]
        # Buy at open day after the signal date
        if in_trade and not first_day_of_trade.empty:
            daily_ret = ticker.loc[current_date, f'{TICKER}_Daily_gain']
            capital *= (1 + daily_ret)
            print(f"  In trade: applied {TICKER} return {daily_ret:.4f}, new capital={capital:.2f}")
            equity.append((current_date, capital))

        elif in_trade:
            # Apply ticker daily return while trade is active
            daily_ret = ticker.loc[current_date, f'{TICKER}_Return_close']
            capital *= (1 + daily_ret)
            print(f"  In trade: applied {TICKER} return {daily_ret:.4f}, new capital={capital:.2f}")

            equity.append((current_date, capital))

            if current_date == exit_date:
                print(f"  >>> EXIT trade on {current_date.date()} at capital={capital:.2f}")
                in_trade = False
                reenter_spy_next_open = True  # next day we must re-enter SPY at open

        else:
            if reenter_spy_next_open:
                # Re-enter SPY at open on the first day after trade exit
                intraday_ret = base.loc[current_date, 'Close'] / base.loc[current_date, 'Open'] - 1
                capital *= (1 + intraday_ret)
                print(f"  Re-entered SPY: open→close return {intraday_ret:.4f}, new capital={capital:.2f}")
                equity.append((current_date, capital))
                reenter_spy_next_open = False

            else:
                # Check if this is a trade entry date
                entry_trades = trades_df.loc[trades_df['Signal_Date'] == current_date]
                if not entry_trades.empty:
                    trade = entry_trades.iloc[0]
                    entry_date, exit_date = trade['Signal_Date'], trade['Exit_Date']

                    # Apply SPY overnight return (close→open) before switching into ticker
                    overnight_ret = base.loc[current_date, 'SPY_Return_overnight']
                    capital *= (1 + overnight_ret)
                    print(f"  Entry signal: SPY overnight return {overnight_ret:.4f}, new capital={capital:.2f}")
                    print(f"  >>> ENTER trade {TICKER} on the next open after {entry_date} until {exit_date.date()}")

                    equity.append((current_date, round(capital, 2)))
                    in_trade = True
                else:
                    # Normal SPY hold (close→close)
                    daily_ret = base.loc[current_date, 'SPY_Return_close']
                    capital *= (1 + daily_ret)
                    print(f"  Hold SPY: applied return {daily_ret:.4f}, new capital={capital:.2f}")

                    equity.append((current_date, round(capital, 2)))
    
    # Convert to Series
    equity_series = pd.Series([v for _, v in equity],
                              index=[d for d, _ in equity])

    if plot:
        spy_equity = (1 + base['SPY_Return_close']).cumprod() * initial_capital
        ticker_equity = (1 + ticker[f'{TICKER}_Return_close']).cumprod() * initial_capital
        plt.figure(figsize=(10, 5))
        plt.plot(equity_series.index, equity_series.values, label=f"{BASE} + {TICKER} Gap Strategy")
        plt.plot(spy_equity.index, spy_equity.values, label=f"{BASE} Buy & Hold")
        plt.plot(ticker_equity.index, ticker_equity.values, label=f"{TICKER} Buy & Hold")
        plt.title("Equity Curve Comparison")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return equity_series

df = get_data(TICKER)
trades, stats = get_gap_trade_details(df, z=2.0, horizon=3)

# print(f"Returned trades: {len(trades)}")
# print(f"Stats n_trades: {stats['n_trades']}")
# print(f"Avg PnL: {trades['Profit_and_loss_pct'].mean():.2f}%")
# print(f"Total PnL: {(1 + trades['Profit_and_loss_pct'] / 100).prod() - 1:.2%}")

equity_curve = generate_equity_curve(trades, initial_capital=10000,plot=True)

# for i in range(0,4):
#     stats_i_days = gap_win_rates(df, z=2.0, horizon=i)
#     print("Day",i)
#     print("Win rate for big gap up", stats_i_days["big_gap_up"]["win_rate"])
#     print("Win rate for big gap down", stats_i_days["big_gap_down"]["win_rate"])
#     print("Win rate for baseline", stats_i_days["baseline_all_days"]["win_rate"])
#     print("P&L for big gap up", stats_i_days["big_gap_up"]["p&L"])
#     print("P&L for big gap down", stats_i_days["big_gap_down"]["p&L"])
df = get_data(BASE)
plot_gap_win_rates_vs_baseline(df, z=2.0, max_horizon=3)

