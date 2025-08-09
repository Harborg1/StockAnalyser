import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

START_DATE = "2020-01-01"
END_DATE = "2025-08-01"
TICKER = "CLSK"

def get_data(ticker=TICKER, start=START_DATE, end=END_DATE):
    data = yf.download(ticker, start=start, end=end, auto_adjust=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data['3Day_Change'] = data['Close'].pct_change(periods=3) * 100
    return data

def backtest_buy_down_sell_up(data, buy_thresh=-10, sell_gain=10, initial_capital=10000, cooldown=3, stop_loss=-10):
    in_position = False
    capital = initial_capital
    shares = 0
    trades = []
    buy_price = 0
    last_trade_index = -cooldown
    last_sell_index = -1000

    for i in range(3, len(data)):
        change_3day = float(data.iloc[i]['3Day_Change'])
        date = data.index[i]
        close = float(data.iloc[i]['Close'])

        # Sell logic
        if in_position:
            holding_change = (close - buy_price) / buy_price * 100
            if change_3day >= sell_gain or holding_change <= stop_loss:
                capital += shares * close
                net_value = capital  # after selling
                trades.append({
                    "Date": date,
                    "Action": "SELL",
                    "Price": close,
                    "Shares": shares,
                    "Capital": round(capital, 2),
                    "Net_Value": round(net_value, 2)
                })
                in_position = False
                last_trade_index = i
                last_sell_index = i

        # Buy logic
        elif (change_3day <= buy_thresh and i - last_trade_index >= cooldown and i - last_sell_index >= cooldown):
            shares = capital // close
            if shares == 0:
                continue
            buy_price = close
            capital -= shares * buy_price
            net_value = capital + shares * buy_price  # equals original capital
            trades.append({
                "Date": date,
                "Action": "BUY",
                "Price": buy_price,
                "Shares": shares,
                "Capital": round(capital, 2),
                "Net_Value": round(net_value, 2)
            })
            in_position = True
            last_trade_index = i

    # Final value of unsold shares
    if in_position:
        final_close = float(data.iloc[-1]['Close'])
        capital += shares * final_close
        net_value = capital
        trades.append({
            "Date": data.index[-1],
            "Action": "Final Value (Unrealized Sell)",
            "Price": final_close,
            "Shares": shares,
            "Capital": round(capital, 2),
            "Net_Value": round(net_value, 2)
        })

    return trades, capital

def summarize_trades(trades, start_capital):
    profits = []
    for i in range(0, len(trades)-1, 2):
        buy = trades[i]['Price']
        sell = trades[i+1]['Price']
        profit = (sell - buy) / buy * 100
        profits.append(profit)

    final_capital = trades[-1]['Capital'] if trades else start_capital
    return {
        "Total Trades": len(profits),
        "Average Return per Trade (%)": round(sum(profits)/len(profits), 2) if profits else 0,
        "Total Return (%)": round(((final_capital - start_capital) / start_capital) * 100, 2),
        "Final Capital ($)": round(final_capital, 2)
    }

# Example usage
start_capital = 10000
data = get_data()
trades, final_cap = backtest_buy_down_sell_up(data, initial_capital=start_capital)
summary = summarize_trades(trades, start_capital)

print(pd.DataFrame(trades))
print(summary)

# Buy and Hold value
start_price = data.loc[data.index[0], "Close"]
end_price = data.loc[data.index[-1], "Close"]
print(start_price)
print(end_price)
buy_and_hold_value = start_capital * (end_price / start_price)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(pd.DataFrame(trades)["Date"], pd.DataFrame(trades)["Net_Value"], label="Strategy (Buy Dip / Sell Rip)")
plt.axhline(y=buy_and_hold_value, color='r', linestyle='--', label="Buy & Hold Final Value")
plt.title(f"{TICKER} Strategy vs Buy and Hold\n({START_DATE} to {END_DATE})")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
