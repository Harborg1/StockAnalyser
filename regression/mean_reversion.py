import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
output_path = "csv_files/mean_reversion_results.csv"
BASE = "SPY"
TICKER = 'TSLA'
INTERVAL = '1d'
PERIOD = '730d' if INTERVAL == '1h' else 'max'
LOOKBACK = 1500
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

def calculate_subset_metrics(valid: pd.DataFrame, mask: pd.Series, is_baseline:bool) -> dict:
    subset = valid.loc[mask]
    if subset.empty:
        return {'count': 0, 'win_rate': np.nan, 'avg_ret_%': np.nan, 'p&L': np.nan}
    
    wins = (subset['Future_Close'] > subset['Open']).mean()

    if is_baseline:
        wins = (subset['Close'] > subset['Open']).mean()

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
    res_up = calculate_subset_metrics(valid, valid['Big_Gap'] == 1,False)
    print("res_up", res_up["win_rate"])
    res_down = calculate_subset_metrics(valid, valid['Big_Gap'] == -1, False)
    print("res_down",res_down["win_rate"])
    base = calculate_subset_metrics(valid, valid['Big_Gap'].isin([-1, 0, 1]), True)
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

    t.to_csv(output_path)

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

    return t,stats


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
        # Load SPY and TSLA
        base = get_data(f'{BASE}')
        ticker = get_data(f'{TICKER}')

        for df in (base, ticker):
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

        # Add daily returns
        base['SPY_Return_close'] = base['Close'].pct_change().fillna(0)
        base['SPY_Return_overnight'] = base['Open'] / base['Close'].shift(1) - 1
        ticker['TSLA_Return_close'] = ticker['Close'].pct_change().fillna(0)

        # Align dates
        all_days = base.index.union(ticker.index).sort_values()
        capital = initial_capital
        equity = []

        in_trade = False
        exit_date = None

        for current_date in all_days:
            if current_date not in base.index or current_date not in ticker.index:
                continue  # skip non-trading days
            if in_trade:
                if current_date == exit_date:
                    # Apply whole ticker trade return at exit
                    trade = trades_df.loc[trades_df['Exit_Date'] == current_date].iloc[0]
                    capital *= (1 + trade['Profit_and_loss_pct'] / 100.0)
                    equity.append((current_date, capital))
                    in_trade = False
                else:
                    # During trade window, equity stays flat
                    equity.append((current_date, capital))
            else:
                # Check if this is a ticker trade entry
                entry_trades = trades_df.loc[trades_df['Signal_Date'] == current_date]
                if not entry_trades.empty:
                    trade = entry_trades.iloc[0]
                    entry_date, exit_date = trade['Signal_Date'], trade['Exit_Date']

                    # Apply SPY overnight return (close→open)
                    capital *= (1 + base.loc[current_date, 'SPY_Return_overnight'])
                    equity.append((current_date, round(capital,2)))

                    in_trade = True
                else:
                    # Normal SPY hold (close→close)
                    capital *= (1 + base.loc[current_date, 'SPY_Return_close'])
                    equity.append((current_date, round(capital,2)))
        print(equity)
        # Convert to Series
        equity_series = pd.Series([v for _, v in equity],
                                index=[d for d, _ in equity])

        if plot:
            # SPY buy & hold
            spy_equity = (1 + base['SPY_Return_close']).cumprod() * initial_capital

            plt.figure(figsize=(10, 5))
            plt.plot(equity_series.index, equity_series.values, label= f"{BASE} + {TICKER} Gap Strategy")
            plt.plot(spy_equity.index, spy_equity.values, label=f"{BASE} Buy & Hold")
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

print(f"Returned trades: {len(trades)}")
print(f"Stats n_trades: {stats['n_trades']}")
print(f"Avg PnL: {trades['Profit_and_loss_pct'].mean():.2f}%")
print(f"Total PnL: {(1 + trades['Profit_and_loss_pct'] / 100).prod() - 1:.2%}")

equity_curve = generate_equity_curve(trades, initial_capital=10000,plot=True)

# for i in range(0,4):
#     stats_i_days = gap_win_rates(df, z=2.0, horizon=i)
#     print("Day",i)
#     print("Win rate for big gap up", stats_i_days["big_gap_up"]["win_rate"])
#     print("Win rate for big gap down", stats_i_days["big_gap_down"]["win_rate"])
#     print("Win rate for baseline", stats_i_days["baseline_all_days"]["win_rate"])
#     print("P&L for big gap up", stats_i_days["big_gap_up"]["p&L"])
#     print("P&L for big gap down", stats_i_days["big_gap_down"]["p&L"])

# plot_gap_win_rates_vs_baseline(df, z=2.0, max_horizon=10)
