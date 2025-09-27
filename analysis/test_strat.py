import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

"""This script compares two trading strategies against the buy-and-hold approach for a given stock.
    The strategies are:
    1. Buy the Dip, Sell the Rip: Buys when the stock price drops by 10% or more over 3 trading days and sells after a 10% gain or stop-loss.
    2. Dollar Cost Averaging (DCA): Invests a fixed amount at regular intervals regardless of price.
    The performance of each strategy is evaluated and plotted.
"""
START_DATE = "2024-01-01"
END_DATE   = "2025-09-27"
TICKER     = "CLSK"


def get_data(ticker=TICKER, start=START_DATE, end=END_DATE):
    data = yf.download(ticker, start=start, end=end, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data["3Day_Change"] = data["Close"].pct_change(periods=3) * 100
    return data

def backtest_strategy(
    data,
    buy_thresh,
    sell_gain,
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
                shares = 0

        # --- Buy logic ---
        elif (change_3day <= buy_thresh and i - last_trade_index >= cooldown):
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


def DCA_strategy(
    data,
    initial_capital=10000,
    fee_bps_per_side=10,
    slippage_bps_per_side=5
):
    portfolio_values = []
    trades = []
    capital = initial_capital
    total_shares = 0

    rt_cost_pct = 2 * (fee_bps_per_side + slippage_bps_per_side) / 1e4

    # Determine number of DCA intervals (e.g., every 5 trading days)
    num_trading_days = len(data)
    num_dca_trades = num_trading_days // 5  # invest every 5 trading days
    investment_interval = 5

    # Recalculate the exact amount to invest per interval
    investment_amount = initial_capital / num_dca_trades

    for i in range(0, num_trading_days, investment_interval):
        date = data.index[i]
        close = float(data.iloc[i]["Close"])

        # Adjusted price including slippage and fees
        price_with_costs = close * (1 + rt_cost_pct)
        num_shares = investment_amount / price_with_costs  # fractional shares

        if num_shares > 0:
            cost = num_shares * price_with_costs
            capital -= cost
            total_shares += num_shares
            trades.append({
                "Date": date,
                "Action": "BUY",
                "Price": close,
                "Shares": round(total_shares, 6),
                "Capital": round(capital, 2)
            })

        # Track daily portfolio value (mark-to-market)
        portfolio_value = capital + total_shares * close
        portfolio_values.append((date, portfolio_value))

    # Fill in remaining dates for complete curve
    last_known_index = 0
    for i in range(num_trading_days):
        date = data.index[i]
        close = float(data.iloc[i]["Close"])
        portfolio_value = capital + total_shares * close
        if i > last_known_index and date > portfolio_values[-1][0]:
            portfolio_values.append((date, portfolio_value))

    final_value = capital + total_shares * float(data.iloc[-1]["Close"])
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

start_capital = 10000
data = get_data()

# Run "buy the dip, sell the rip" strategy
trades_bsr, final_cap_bsr, equity_curve_bsr = backtest_strategy(
    data, buy_thresh=-10,sell_gain=10, initial_capital=start_capital
)

# Make dataframe og transactions 
trades_bsr_df = pd.DataFrame(trades_bsr)
print("\nAlle handler (Buy Down / Sell Up):")
print(trades_bsr_df)


# Run FOMO strategy
trades_FOMO, final_cap_bsr, equity_curve_FOMO = backtest_strategy(
    data, buy_thresh=10,sell_gain=-10, initial_capital=start_capital
)



# Run DCA strategy
trades_dca, final_cap_dca, equity_curve_dca = DCA_strategy(
    data, initial_capital=start_capital
)


# Buy & Hold strategy (hele kapitalen investeret fra start)
shares_bh = start_capital / data.iloc[0]["Close"]
equity_curve_bh = pd.DataFrame({
    "Date": data.index,
    "Portfolio": shares_bh * data["Close"]
})

# Summaries
summary_bsr = summarize_trades(trades_bsr, start_capital, data)

summary_FOMO = summarize_trades(trades_FOMO, start_capital, data)

summary_dca = summarize_trades(trades_dca, start_capital, data)

print("Buy-Sell-Rip Summary:")
print(summary_bsr)
print("FOMO summary")
print(summary_FOMO)
print("\nDCA Summary:")
print(summary_dca)

# ==== Align all equity curves to daily data ====
equity_curve_bsr = (
    equity_curve_bsr.drop_duplicates("Date")
    .set_index("Date")
    .reindex(data.index, method="ffill")
)


equity_curve_FOMO = (
    equity_curve_FOMO.drop_duplicates("Date")
    .set_index("Date")
    .reindex(data.index, method="ffill")
)


equity_curve_dca = (
    equity_curve_dca.drop_duplicates("Date")
    .set_index("Date")
    .reindex(data.index, method="ffill")
)

equity_curve_bh = (
    equity_curve_bh.drop_duplicates("Date")
    .set_index("Date")
)

# ==== Plot all equity curves ====
plt.figure(figsize=(12, 6))
plt.plot(equity_curve_bsr.index, equity_curve_bsr["Portfolio"], label="Buy Down / Sell Up")
plt.plot(equity_curve_FOMO.index, equity_curve_FOMO["Portfolio"], label="Buy Up / Sell Down")
#plt.plot(equity_curve_dca.index, equity_curve_dca["Portfolio"], label="DCA")
plt.plot(equity_curve_bh.index, equity_curve_bh["Portfolio"], label="Buy & Hold")

plt.title(f"{TICKER} Strategies vs Buy & Hold\n({START_DATE} to {END_DATE})")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

