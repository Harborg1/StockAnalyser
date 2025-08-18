import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

START_DATE = "2024-01-01"
END_DATE   = "2025-08-01"
TICKER     = "CLSK"

def get_data(ticker=TICKER, start=START_DATE, end=END_DATE):
    data = yf.download(ticker, start=start, end=end, auto_adjust=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data["3Day_Change"] = data["Close"].pct_change(periods=3) * 100
    return data


def backtest_buy_down_sell_up(
    data,
    buy_thresh=-10,
    sell_gain=10,
    initial_capital=10000,
    cooldown=3,
    stop_loss=-10,
    fee_bps_per_side=10,   # 0.1% cost
    slippage_bps_per_side=5  # 0.05% slippage
):
    in_position = False
    capital = initial_capital
    shares = 0
    trades = []
    buy_price = 0
    last_trade_index = -cooldown
    last_sell_index = -1000

    # For equity curve tracking
    portfolio_values = []

    # Round trip cost (both sides)
    rt_cost_pct = 2 * (fee_bps_per_side + slippage_bps_per_side) / 1e4

    for i in range(3, len(data) - 1):  # -1 so we always have "next bar" available
        change_3day = float(data.iloc[i]["3Day_Change"])
        date = data.index[i]
        close = float(data.iloc[i]["Close"])
        next_open = float(data.iloc[i + 1]["Open"])  # execute at NEXT open

        # Track daily portfolio value (mark-to-market)
        current_value = capital + shares * close
        portfolio_values.append((date, current_value))

        # --- Sell logic ---
        if in_position:
            holding_change = (close - buy_price) / buy_price * 100
            if change_3day >= sell_gain or holding_change <= stop_loss:
                capital += shares * next_open * (1 - rt_cost_pct)
                trades.append({
                    "Date": data.index[i + 1],
                    "Action": "SELL",
                    "Price": next_open,
                    "Shares": shares,
                    "Capital": round(capital, 2)
                })
                in_position = False
                last_trade_index = i
                last_sell_index = i
                shares = 0

        # --- Buy logic ---
        elif (change_3day <= buy_thresh and i - last_trade_index >= cooldown and i - last_sell_index >= cooldown):
            shares = capital // (next_open * (1 + rt_cost_pct))
            if shares > 0:
                cost = shares * next_open * (1 + rt_cost_pct)
                buy_price = next_open
                capital -= cost
                trades.append({
                    "Date": data.index[i + 1],
                    "Action": "BUY",
                    "Price": buy_price,
                    "Shares": shares,
                    "Capital": round(capital, 2)
                })
                in_position = True
                last_trade_index = i

    # Final value if still holding
    final_close = float(data.iloc[-1]["Close"])
    final_value = capital + shares * final_close
    portfolio_values.append((data.index[-1], final_value))

    return trades, final_value, pd.DataFrame(portfolio_values, columns=["Date", "Portfolio"])

def summarize_trades(trades, start_capital, data=None):
    profits = []
    for i in range(0, len(trades) - 1, 2):
        buy = trades[i]["Price"]
        sell = trades[i + 1]["Price"]
        profit = (sell - buy) / buy * 100
        profits.append(profit)

    if not trades:
        final_capital = start_capital
    else:
        last_trade = trades[-1]
        if last_trade["Action"] == "SELL":
            final_capital = last_trade["Capital"]
        else:  # still holding
            final_close = data.iloc[-1]["Close"]
            final_capital = last_trade["Capital"] + last_trade["Shares"] * final_close

    return {
        "Total Trades": len(profits),
        "Average Return per Trade (%)": float(round(sum(profits) / len(profits), 2)) if profits else 0,
        "Total Return (%)": float(round(((final_capital - start_capital) / start_capital) * 100, 2)),
        "Final Capital ($)": float(round(final_capital, 2))
    }


# ==== Run example ====
start_capital = 10000
data = get_data()
trades, final_cap, equity_curve = backtest_buy_down_sell_up(data, initial_capital=start_capital)
summary = summarize_trades(trades, start_capital,data)

print(pd.DataFrame(trades))
print(summary)

# Buy & Hold
start_price = data.iloc[0]["Close"]
end_price = data.iloc[-1]["Close"]
buy_and_hold_value = start_capital * (end_price / start_price)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(equity_curve["Date"], equity_curve["Portfolio"], label="Strategy")
plt.axhline(y=buy_and_hold_value, color="r", linestyle="--", label="Buy & Hold Final Value")
plt.title(f"{TICKER} Strategy vs Buy & Hold\n({START_DATE} to {END_DATE})")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
