"""Compare simple trading strategies with a buy-and-hold benchmark.

Signals are calculated using a day's closing price and executed at the next
trading day's open. Portfolio values are marked at each day's close.
"""

#https://www.sec.gov/edgar/search/#/dateRange=10y&category=form-cat1&ciks=0001652044&entityName=Alphabet%2520Inc.%2520(GOOG%252C%2520GOOGL)%2520(CIK%25200001652044)


from math import floor, sqrt

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


START_DATE = "2024-01-01"
END_DATE = "2026-08-01"
TICKER = "CLSK"
INITIAL_CAPITAL = 10_000.0
FEE_BPS_PER_SIDE = 10
SLIPPAGE_BPS_PER_SIDE = 5
ALLOW_FRACTIONAL_SHARES = True


def get_data(ticker=TICKER, start=START_DATE, end=END_DATE):
    """Download split- and dividend-adjusted OHLC data for one ticker."""
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = _prepare_data(data)
    data["3Day_Change"] = data["Close"].pct_change(periods=3) * 100
    return data


def _prepare_data(data):
    """Validate and normalize price data used by every strategy."""
    if data is None or data.empty:
        raise ValueError("Price data is empty.")

    required = {"Open", "Close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Price data is missing columns: {sorted(missing)}")

    clean = data.copy().sort_index()
    clean = clean.loc[~clean.index.duplicated(keep="last")]
    for column in required:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=list(required))

    if clean.empty:
        raise ValueError("Price data has no usable Open and Close rows.")
    if (clean[list(required)] <= 0).any().any():
        raise ValueError("Open and Close prices must be positive.")

    if "3Day_Change" not in clean:
        clean["3Day_Change"] = clean["Close"].pct_change(periods=3) * 100
    else:
        clean["3Day_Change"] = pd.to_numeric(
            clean["3Day_Change"], errors="coerce"
        )
    return clean


def _side_cost_rate(fee_bps_per_side, slippage_bps_per_side):
    if fee_bps_per_side < 0 or slippage_bps_per_side < 0:
        raise ValueError("Fees and slippage cannot be negative.")
    return (fee_bps_per_side + slippage_bps_per_side) / 10_000


def _share_quantity(budget, price, side_cost, allow_fractional_shares):
    quantity = budget / (price * (1 + side_cost))
    return quantity if allow_fractional_shares else float(floor(quantity))


def _run_signal_strategy(
    data,
    buy_signal,
    sell_signal,
    initial_capital,
    cooldown,
    fee_bps_per_side,
    slippage_bps_per_side,
    allow_fractional_shares,
):
    """Run close-generated signals using next-open execution."""
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    if cooldown < 0:
        raise ValueError("Cooldown cannot be negative.")

    data = _prepare_data(data)
    side_cost = _side_cost_rate(fee_bps_per_side, slippage_bps_per_side)
    cash = float(initial_capital)
    shares = 0.0
    entry_price = None
    entry_total_cost = None
    pending_action = None
    last_exit_index = -cooldown
    trades = []
    portfolio_values = []

    for i, (date, row) in enumerate(data.iterrows()):
        open_price = float(row["Open"])
        close_price = float(row["Close"])

        if pending_action == "BUY" and shares == 0:
            quantity = _share_quantity(
                cash, open_price, side_cost, allow_fractional_shares
            )
            if quantity > 0:
                gross_cost = quantity * open_price
                costs = gross_cost * side_cost
                total_cost = gross_cost + costs
                cash = max(0.0, cash - total_cost)
                shares = quantity
                entry_price = open_price
                entry_total_cost = total_cost
                trades.append(
                    {
                        "Date": date,
                        "Action": "BUY",
                        "Price": open_price,
                        "Shares": quantity,
                        "Costs": costs,
                        "Capital": cash,
                    }
                )

        elif pending_action == "SELL" and shares > 0:
            quantity = shares
            gross_proceeds = quantity * open_price
            costs = gross_proceeds * side_cost
            net_proceeds = gross_proceeds - costs
            net_profit = net_proceeds - entry_total_cost
            net_return = net_profit / entry_total_cost * 100
            cash += net_proceeds
            trades.append(
                {
                    "Date": date,
                    "Action": "SELL",
                    "Price": open_price,
                    "Shares": quantity,
                    "Costs": costs,
                    "Net Profit": net_profit,
                    "Return (%)": net_return,
                    "Capital": cash,
                }
            )
            shares = 0.0
            entry_price = None
            entry_total_cost = None
            last_exit_index = i

        portfolio_values.append((date, cash + shares * close_price))
        pending_action = None

        if i == len(data) - 1:
            continue

        change_3day = row["3Day_Change"]
        if pd.isna(change_3day):
            continue
        change_3day = float(change_3day)

        if shares > 0:
            holding_return = (close_price - entry_price) / entry_price * 100
            if sell_signal(change_3day, holding_return):
                pending_action = "SELL"
        elif (i + 1 - last_exit_index) >= cooldown and buy_signal(change_3day):
            pending_action = "BUY"

    equity_curve = pd.DataFrame(portfolio_values, columns=["Date", "Portfolio"])
    final_value = float(equity_curve["Portfolio"].iloc[-1])
    return trades, final_value, equity_curve


def backtest_strategy(
    data,
    buy_thresh=-10,
    sell_gain=10,
    initial_capital=INITIAL_CAPITAL,
    cooldown=3,
    stop_loss=-10,
    fee_bps_per_side=FEE_BPS_PER_SIDE,
    slippage_bps_per_side=SLIPPAGE_BPS_PER_SIDE,
    allow_fractional_shares=ALLOW_FRACTIONAL_SHARES,
):
    """Buy a three-day dip, then exit on gain from entry or stop loss."""
    if buy_thresh >= 0:
        raise ValueError("The dip strategy buy threshold must be negative.")
    if sell_gain <= 0:
        raise ValueError("The dip strategy sell gain must be positive.")
    if stop_loss >= 0:
        raise ValueError("Stop loss must be negative.")

    return _run_signal_strategy(
        data=data,
        buy_signal=lambda change: change <= buy_thresh,
        sell_signal=lambda _change, holding_return: (
            holding_return >= sell_gain or holding_return <= stop_loss
        ),
        initial_capital=initial_capital,
        cooldown=cooldown,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
        allow_fractional_shares=allow_fractional_shares,
    )


def backtest_fomo_strategy(
    data,
    buy_thresh=10,
    reversal_thresh=-10,
    stop_loss=-10,
    initial_capital=INITIAL_CAPITAL,
    cooldown=3,
    fee_bps_per_side=FEE_BPS_PER_SIDE,
    slippage_bps_per_side=SLIPPAGE_BPS_PER_SIDE,
    allow_fractional_shares=ALLOW_FRACTIONAL_SHARES,
):
    """Buy strong three-day momentum and exit on reversal or stop loss."""
    if buy_thresh <= 0:
        raise ValueError("The FOMO buy threshold must be positive.")
    if reversal_thresh >= 0 or stop_loss >= 0:
        raise ValueError("FOMO reversal and stop-loss thresholds must be negative.")

    return _run_signal_strategy(
        data=data,
        buy_signal=lambda change: change >= buy_thresh,
        sell_signal=lambda change, holding_return: (
            change <= reversal_thresh or holding_return <= stop_loss
        ),
        initial_capital=initial_capital,
        cooldown=cooldown,
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
        allow_fractional_shares=allow_fractional_shares,
    )


def DCA_strategy(
    data,
    initial_capital=INITIAL_CAPITAL,
    investment_interval=5,
    fee_bps_per_side=FEE_BPS_PER_SIDE,
    slippage_bps_per_side=SLIPPAGE_BPS_PER_SIDE,
    allow_fractional_shares=ALLOW_FRACTIONAL_SHARES,
):
    """Gradually deploy an existing lump sum at regular trading-day intervals."""
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    if not isinstance(investment_interval, int) or investment_interval <= 0:
        raise ValueError("Investment interval must be a positive integer.")

    data = _prepare_data(data)
    side_cost = _side_cost_rate(fee_bps_per_side, slippage_bps_per_side)
    purchase_indices = list(range(0, len(data), investment_interval))
    investment_amount = initial_capital / len(purchase_indices)
    purchase_set = set(purchase_indices)
    cash = float(initial_capital)
    total_shares = 0.0
    trades = []
    portfolio_values = []

    for i, (date, row) in enumerate(data.iterrows()):
        open_price = float(row["Open"])
        close_price = float(row["Close"])

        if i in purchase_set and cash > 0:
            is_last_purchase = i == purchase_indices[-1]
            budget = cash if is_last_purchase else min(investment_amount, cash)
            quantity = _share_quantity(
                budget, open_price, side_cost, allow_fractional_shares
            )
            if quantity > 0:
                gross_cost = quantity * open_price
                costs = gross_cost * side_cost
                total_cost = min(gross_cost + costs, cash)
                cash = max(0.0, cash - total_cost)
                total_shares += quantity
                trades.append(
                    {
                        "Date": date,
                        "Action": "BUY",
                        "Price": open_price,
                        "Shares": quantity,
                        "Total Shares": total_shares,
                        "Costs": costs,
                        "Capital": cash,
                    }
                )

        portfolio_values.append((date, cash + total_shares * close_price))

    equity_curve = pd.DataFrame(portfolio_values, columns=["Date", "Portfolio"])
    final_value = float(equity_curve["Portfolio"].iloc[-1])
    return trades, final_value, equity_curve


def buy_and_hold_strategy(
    data,
    initial_capital=INITIAL_CAPITAL,
    fee_bps_per_side=FEE_BPS_PER_SIDE,
    slippage_bps_per_side=SLIPPAGE_BPS_PER_SIDE,
    allow_fractional_shares=ALLOW_FRACTIONAL_SHARES,
):
    """Invest at the first open using the same cost and share assumptions."""
    data = _prepare_data(data)
    side_cost = _side_cost_rate(fee_bps_per_side, slippage_bps_per_side)
    first_open = float(data["Open"].iloc[0])
    shares = _share_quantity(
        initial_capital, first_open, side_cost, allow_fractional_shares
    )
    gross_cost = shares * first_open
    costs = gross_cost * side_cost
    cash = max(0.0, initial_capital - gross_cost - costs)
    equity_curve = pd.DataFrame(
        {
            "Date": data.index,
            "Portfolio": cash + shares * data["Close"].to_numpy(),
        }
    )
    trade = {
        "Date": data.index[0],
        "Action": "BUY",
        "Price": first_open,
        "Shares": shares,
        "Costs": costs,
        "Capital": cash,
    }
    return [trade], float(equity_curve["Portfolio"].iloc[-1]), equity_curve


def summarize_trades(trades, start_capital, final_value, equity_curve):
    """Summarize portfolio performance and completed round trips."""
    if start_capital <= 0:
        raise ValueError("Start capital must be positive.")
    if equity_curve is None or equity_curve.empty:
        raise ValueError("An equity curve is required for the summary.")

    completed = [trade for trade in trades if trade["Action"] == "SELL"]
    net_returns = [float(trade["Return (%)"]) for trade in completed]
    portfolio = equity_curve.set_index("Date")["Portfolio"].astype(float)
    daily_returns = portfolio.pct_change().dropna()
    running_peak = portfolio.cummax()
    max_drawdown = float(((portfolio / running_peak) - 1).min() * 100)

    elapsed_days = (portfolio.index[-1] - portfolio.index[0]).days
    cagr = None
    if elapsed_days > 0 and final_value > 0:
        cagr = ((final_value / start_capital) ** (365.25 / elapsed_days) - 1) * 100

    volatility = float(daily_returns.std(ddof=1) * sqrt(252) * 100)
    if len(daily_returns) < 2 or pd.isna(volatility):
        volatility = None

    sharpe = None
    daily_std = daily_returns.std(ddof=1)
    if len(daily_returns) >= 2 and pd.notna(daily_std) and daily_std > 0:
        sharpe = float(daily_returns.mean() / daily_std * sqrt(252))

    return {
        "Purchases": sum(trade["Action"] == "BUY" for trade in trades),
        "Completed Trades": len(completed),
        "Win Rate (%)": (
            round(sum(value > 0 for value in net_returns) / len(net_returns) * 100, 2)
            if net_returns
            else None
        ),
        "Average Net Trade Return (%)": (
            round(sum(net_returns) / len(net_returns), 2) if net_returns else None
        ),
        "Total Return (%)": round((final_value / start_capital - 1) * 100, 2),
        "CAGR (%)": round(cagr, 2) if cagr is not None else None,
        "Maximum Drawdown (%)": round(max_drawdown, 2),
        "Annualized Volatility (%)": (
            round(volatility, 2) if volatility is not None else None
        ),
        "Sharpe Ratio (0% RF)": round(sharpe, 2) if sharpe is not None else None,
        "Final Capital ($)": round(final_value, 2),
    }


def plot_equity_curves(curves, ticker=TICKER, start=START_DATE, end=END_DATE):
    plt.figure(figsize=(12, 6))
    for label, curve in curves.items():
        plt.plot(curve["Date"], curve["Portfolio"], label=label)
    plt.title(f"{ticker} Strategies vs Buy & Hold\n({start} to {end})")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    data = get_data()
    trades_bsr, final_bsr, curve_bsr = backtest_strategy(data)
    trades_fomo, final_fomo, curve_fomo = backtest_fomo_strategy(data)
    trades_dca, final_dca, curve_dca = DCA_strategy(data)
    trades_bh, final_bh, curve_bh = buy_and_hold_strategy(data)

    results = {
        "Buy Down / Sell Up": (trades_bsr, final_bsr, curve_bsr),
        "Momentum / Reversal": (trades_fomo, final_fomo, curve_fomo),
        "Staged Lump Sum": (trades_dca, final_dca, curve_dca),
        "Buy & Hold": (trades_bh, final_bh, curve_bh),
    }

    for name, (trades, final_value, curve) in results.items():
        print(f"\n{name} transactions:")
        print(pd.DataFrame(trades))
        print(f"{name} summary:")
        print(summarize_trades(trades, INITIAL_CAPITAL, final_value, curve))

    plot_equity_curves(
        {name: result[2] for name, result in results.items()},
        ticker=TICKER,
        start=START_DATE,
        end=END_DATE,
    )


if __name__ == "__main__":
    main()
