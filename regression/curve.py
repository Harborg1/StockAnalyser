import yfinance as yf
import matplotlib.pyplot as plt

def plot_buy_and_hold_equity_curve(ticker: str, start_date: str, end_date: str, initial_capital: float = 10000.0):

    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    close = data['Close']

    returns = close.pct_change().fillna(0)
    equity_curve = (1 + returns).cumprod() * initial_capital

    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve.index, equity_curve.values)
    plt.title(f"{ticker} Buy & Hold Equity Curve\n{start_date} to {end_date}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return equity_curve

if __name__=="main":
    plot_buy_and_hold_equity_curve("SPY","2008-04-08","2025-01-08")
