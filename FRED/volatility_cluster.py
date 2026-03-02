import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
import warnings

# Mute warnings for a cleaner output
warnings.filterwarnings('ignore')

# 1. Download Data (Using multi_level_index=False to prevent KeyError)
ticker = "^GSPC"
print("Downloading S&P 500 and VIX data...")
data = yf.download(ticker, start="2006-01-01", end="2026-02-19", multi_level_index=False)
vix_data = yf.download("^VIX", start="2006-01-01", end="2026-02-19", multi_level_index=False)

# 2. Calculate Daily Log Returns
data['Return'] = 100 * np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)

# 3. Fit a REALISTIC Model: EGARCH(1,1) with Student's t-distribution
# p=1: Clustering | o=1: Asymmetry (Leverage effect) | q=1: Shocks
# dist='t': Accounts for fat tails (extreme crashes)
print("Fitting EGARCH model...")
model = arch_model(data['Return'], vol='EGARCH', p=1, o=1, q=1, dist='t')
model_results = model.fit(disp='off')

# 4. Extract and Annualize Conditional Volatility
data['Conditional_Vol'] = model_results.conditional_volatility
# Multiply by sqrt(252) to match the VIX's annualized scale
data['Annualized_GARCH'] = data['Conditional_Vol'] * np.sqrt(252)

# 5. Align Data into a Comparison DataFrame
comparison = pd.DataFrame({
    'GARCH_Annualized': data['Annualized_GARCH'],
    'VIX': vix_data['Close']
}).dropna()

# Calculate Correlation
correlation = comparison.corr().iloc[0, 1]
print(f"Correlation between Realized EGARCH and VIX: {correlation:.2f}")

# 6. Professional Visualizations
fig = plt.figure(figsize=(10,10))

# Plot A: Daily Returns (The Raw Heartbeat)
plt.subplot(3, 1, 1)
plt.plot(data.index, data['Return'], color='gray', alpha=0.5, label='Daily Log Returns')
plt.title('S&P 500 Daily Returns & Volatility Clustering')
plt.ylabel('Returns %')
plt.legend(loc='upper right')

# Plot B: VIX vs EGARCH Over Time (The "Spread" View)
plt.subplot(3, 1, 2)
plt.plot(comparison.index, comparison['VIX'], color='darkorange', label='VIX (Expectation)', alpha=0.8)
plt.plot(comparison.index, comparison['GARCH_Annualized'], color='darkred', label='EGARCH (Reality)', alpha=0.8)
plt.title('Volatility Over Time: Expectation vs. Reality')
plt.ylabel('Annualized Volatility %')
plt.legend(loc='upper right')

# Plot C: Scatter Plot with the Volatility Risk Premium (VRP) Line
plt.subplot(3, 1, 3)
plt.scatter(comparison['VIX'], comparison['GARCH_Annualized'], alpha=0.3, color='purple')

# Add the 45-degree line (1:1 Ratio)
min_val = min(comparison['VIX'].min(), comparison['GARCH_Annualized'].min())
max_val = max(comparison['VIX'].max(), comparison['GARCH_Annualized'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Ratio (Zero Premium)')

plt.title(f"VIX vs. EGARCH Reality Check (Correlation: {correlation:.2f})")
plt.xlabel("VIX (Market Expectation %)")
plt.ylabel("EGARCH (Realized Math %)")
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# 7. Print Model Parameters
print("\n--- EGARCH(1,1) Model Summary ---")
print(model_results.summary())