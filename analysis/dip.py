import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

START_DATE = "2023-01-01" 
END_DATE   = "2025-12-31"
TICKER     = "CLSK"

def calculate_rsi(data, window=14):
    """Standard RSI calculation."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_data(ticker=TICKER, start=START_DATE, end=END_DATE):
    # Fetch data - auto_adjust=False to keep 'Close' and 'Adj Close' distinct if needed
    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    data["Prev_Close"] = data["Close"].shift(1)
    data["RSI"] = calculate_rsi(data)
    
    # --- NEW: Calculate All-Time High (Rolling) ---
    data["ATH"] = data["Close"].rolling(window=252).max()
    
    return data.dropna(subset=["Prev_Close", "RSI", "ATH"])

def backtest_dip_strategy(data, dip_pct=0.10, sell_gain=0.10, stop_loss=-0.6, initial_capital=10000):
    capital = initial_capital
    shares = 0
    in_position = False
    buy_price = 0
    entry_date = None
    entry_rsi = 0
    
    trade_log = []
    portfolio_values = []
    cost_factor = 15 / 10000 

    for i in range(len(data)):
        date = data.index[i]
        low = float(data.iloc[i]["Low"])
        close = float(data.iloc[i]["Close"])
        prev_close = float(data.iloc[i]["Prev_Close"])
        current_rsi = float(data.iloc[i]["RSI"])
        current_ath = float(data.iloc[i]["ATH"])
        
        # 1. SELL LOGIC
        if in_position:
            holding_return = (close - buy_price) / buy_price
            
            if holding_return >= sell_gain or holding_return <= stop_loss:
                sell_price = close * (1 - cost_factor)
                capital = shares * sell_price
                
                pnl = (sell_price - buy_price) / buy_price * 100
                outcome = "SUCCESS ✅" if pnl > 0 else "FAILURE ❌"
                reason = "Target Hit" if holding_return >= sell_gain else "Stop Loss"
                
                trade_log.append({
                    "Entry Date": entry_date.date(),
                    "Exit Date": date.date(),
                    "Buy Price": round(buy_price, 2),
                    "Sell Price": round(close, 2),
                    "Entry RSI": round(entry_rsi, 2),
                    "Return %": round(pnl, 2),
                    "Outcome": outcome,
                    "Reason": reason
                })
                shares = 0
                in_position = False

        # 2. BUY LOGIC (Modified with ATH Filter)
        elif not in_position:
            dip_trigger_price = prev_close * (1 - dip_pct)
            
            # CONDITION: Dip hit AND price is NOT within 20% of ATH
            ath_filter_passed = dip_trigger_price <= (current_ath * 0.80)
            
            if low <= dip_trigger_price and ath_filter_passed:
                buy_price_with_costs = dip_trigger_price * (1 + cost_factor)
                shares = capital // buy_price_with_costs
                if shares > 0:
                    capital -= (shares * buy_price_with_costs)
                    buy_price = dip_trigger_price
                    entry_date = date
                    entry_rsi = current_rsi
                    in_position = True

        current_val = capital + (shares * close)
        portfolio_values.append({"Date": date, "Portfolio": current_val})

    return pd.DataFrame(trade_log), pd.DataFrame(portfolio_values)

# --- Execution ---
data = get_data()
trade_report, equity_curve_dip = backtest_dip_strategy(data)

# --- NEW: Plotting & Visualization ---
# Calculate Buy & Hold baseline starting with the same $10,000
initial_price = data.iloc[0]['Close']
equity_curve_bh = (10000 / initial_price) * data['Close']

plt.figure(figsize=(12, 6))

# Plot Strategy Portfolio Value
plt.plot(equity_curve_dip['Date'], equity_curve_dip['Portfolio'], 
         label=f'Strategy: Dip + ATH Filter ({TICKER})', color='#1f77b4', linewidth=2)

# Plot Buy & Hold Baseline
plt.plot(data.index, equity_curve_bh, 
         label='Buy & Hold Baseline', color='#7f7f7f', linestyle='--', alpha=0.7)

plt.title(f"Portfolio Growth: Strategy vs Buy & Hold ({TICKER})", fontsize=14, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Portfolio Value ($)", fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Show the plot
plt.show()

# --- Output Report (Original) ---
print(f"\n{'='*85}")
print(f"TRADING LOG FOR {TICKER} (DIP + ATH FILTER)")
print(f"{'='*85}")

if trade_report.empty:
    print("No trades were executed. (ATH Filter or Dip criteria not met)")
else:
    cols = ["Entry Date", "Exit Date", "Buy Price", "Sell Price", "Entry RSI", "Return %", "Outcome", "Reason"]
    print(trade_report[cols].to_string(index=False))

print(f"\n{'='*85}")
print(f"FINAL PERFORMANCE SUMMARY")
print(f"{'='*85}")
strategy_final = equity_curve_dip['Portfolio'].iloc[-1]
bh_final = equity_curve_bh.iloc[-1]

print(f"Total Trades:      {len(trade_report)}")
if not trade_report.empty:
    win_rate = (trade_report['Outcome'] == "SUCCESS ✅").sum() / len(trade_report) * 100
    print(f"Win Rate:          {win_rate:.1f}%")

print(f"Final Strategy:    ${strategy_final:,.2f}")
print(f"Final Buy & Hold:  ${bh_final:,.2f}")