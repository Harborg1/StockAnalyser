import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. GENERATE SYNTHETIC DATA (Same as before)
np.random.seed(42)
days = 240 * 2
dates = pd.date_range("2024-01-01", periods=days, freq="B")
daily_returns = np.random.normal(0.002, 0.05, days)
price = 10 * np.exp(np.cumsum(daily_returns))
data = pd.DataFrame({"Close": price}, index=dates)

# FIX 1: Define a consistent logic for Low generation to be used in both original and sim
def generate_lows(close_prices):
    # Using the same random distribution logic for fairness
    return close_prices * (1 - np.random.uniform(0.02, 0.08, len(close_prices)))

data["Low"] = generate_lows(data["Close"])
data["Prev_Close"] = data["Close"].shift(1)
data["ATH"] = data["Close"].rolling(window=252, min_periods=1).max()
data = data.dropna()

def backtest_logic(df, dip_pct=0.10, sell_gain=0.10):
    capital = 10000
    shares = 0
    in_position = False
    buy_price = 0
    cost_factor = 15/10000
    
    for i in range(len(df)):
        # Using .iat for speed (faster than .iloc in loops)
        close = df.iat[i, df.columns.get_loc("Close")]
        low = df.iat[i, df.columns.get_loc("Low")]
        prev_close = df.iat[i, df.columns.get_loc("Prev_Close")]
        ath = df.iat[i, df.columns.get_loc("ATH")]
        
        if in_position:
            ret = (close - buy_price) / buy_price
            if ret >= sell_gain:
                capital = shares * close * (1 - cost_factor)
                shares = 0
                in_position = False
        else:
            trigger = prev_close * (1 - dip_pct)
            # Logic: If Low dropped below trigger, we bought at trigger
            if low <= trigger and trigger <= (ath * 0.80):
                buy_cost = trigger * (1 + cost_factor)
                shares = capital / buy_cost # Allow fractional shares for simplicity
                capital = 0
                buy_price = trigger
                in_position = True
                
    final_value = capital + (shares * df.iloc[-1]["Close"])
    return final_value

# 2. RUN ORIGINAL STRATEGY
strategy_final_val = backtest_logic(data)
strategy_return = (strategy_final_val - 10000) / 10000
market_return = (data.iloc[-1]["Close"] - data.iloc[0]["Close"]) / data.iloc[0]["Close"]

print(f"Original Strategy Return: {strategy_return:.2%}")
print(f"Original Market Return:   {market_return:.2%}")

# 3. MONTE CARLO PERMUTATION (Corrected)
n_simulations = 500
excess_returns = [] # Strategy Return minus Market Return

raw_returns = data["Close"].pct_change().dropna().values

for _ in range(n_simulations):
    np.random.shuffle(raw_returns)
    
    # Reconstruct Price Path
    sim_price = data.iloc[0]["Close"] * np.exp(np.cumsum(np.log(1 + raw_returns)))
    sim_df = pd.DataFrame({"Close": sim_price}, index=data.index[1:])
    
    # FIX 2: Use the SAME logic for Low generation
    sim_df["Low"] = generate_lows(sim_df["Close"])
    
    sim_df["Prev_Close"] = sim_df["Close"].shift(1)
    sim_df["ATH"] = sim_df["Close"].rolling(window=252, min_periods=1).max()
    sim_df = sim_df.dropna()
    
    # Run Strategy
    sim_strat_val = backtest_logic(sim_df)
    sim_strat_ret = (sim_strat_val - 10000) / 10000
    
    # Calculate Market Return for this specific shuffle
    sim_market_ret = (sim_df.iloc[-1]["Close"] - sim_df.iloc[0]["Close"]) / sim_df.iloc[0]["Close"]
    
    # Metric: Did the strategy beat the market IN THIS SIMULATION?
    excess_returns.append(sim_strat_ret - sim_market_ret)

# 4. EVALUATE "BEATING THE MARKET"
# We want to know if the strategy produces positive Alpha (Excess Return) consistently
actual_excess = strategy_return - market_return
p_value_loss = sum(1 for x in excess_returns if x > actual_excess) / n_simulations

plt.figure(figsize=(10, 6))
plt.hist(excess_returns, bins=30, color='lightgreen', edgecolor='black', alpha=0.7, label='Random Excess Returns (Alpha)')
plt.axvline(actual_excess, color='red', linestyle='dashed', linewidth=2, label=f'Actual Alpha ({actual_excess:.1%})')
plt.axvline(0, color='black', linewidth=1, label='Market Performance')
plt.title(f"Can we beat the market? (Alpha Distribution)\nActual Alpha: {actual_excess:.2%}")
plt.xlabel("Strategy Return - Buy & Hold Return")
plt.legend()
plt.tight_layout()
plt.show()