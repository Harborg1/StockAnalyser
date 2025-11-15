import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats

# Download data
ticker = "SPY"
df = yf.download(ticker, start="2015-01-01", end="2015-12-31")
df = df["Close"].dropna()

# Compute log prices and prepare variables
log_price = np.log(df)
x = log_price.shift(1).dropna()
dx = log_price.diff().dropna()
x = x.loc[dx.index]

# Critical: Add time step (daily = 1/252 for annualized parameters)
dt = 1/252  # Assuming daily data

# Prepare arrays - ensure they are 1D
x_vals = np.asarray(x.values, dtype=float).flatten()
dx_vals = np.asarray(dx.values, dtype=float).flatten()

# Manual linear regression using numpy
def linear_regression(x, y):
    # Add column of ones for intercept
    X = np.column_stack([np.ones(len(x)), x])
    
    # Solve using least squares
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    intercept, slope = beta
    
    # Calculate statistics
    y_pred = intercept + slope * x
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (len(x) - 2)
    
    # Standard errors
    XTX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XTX_inv) * mse)
    _, se_slope = se

    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # T-statistics and p-values
    t_slope = slope / se_slope
    p_slope = 2 * (1 - stats.t.cdf(np.abs(t_slope), len(x) - 2))
    
    return float(slope), float(intercept), float(r_squared), float(p_slope), float(se_slope)

# Perform regression
slope, intercept, r_squared, p_value, stderr = linear_regression(x_vals, dx_vals)

# Estimate OU parameters correctly - convert to floats
theta = float(-slope / dt)
mu = float(intercept / (theta * dt)) if theta != 0 else np.nan

# Estimate volatility (annualized)
residuals = dx_vals - (intercept + slope * x_vals)
sigma = float(np.std(residuals) * np.sqrt(1 / dt))

print("Estimated OU parameters:")
print(f"Theta (mean reversion speed): {theta:.4f}")
print(f"Mu (long-term mean):         {mu:.4f}")
print(f"Sigma (volatility):          {sigma:.4f}")
print(f"Half-life (years):           {np.log(2)/theta:.4f}")
print(f"Half-life (trading days):    {np.log(2)/theta/dt:.1f}")
print(f"R-squared:                   {r_squared:.4f}")
print(f"P-value for slope:           {p_value:.4f}")

# Plot results
fitted_drift = intercept + slope * x_vals
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(x_vals, dx_vals, alpha=0.3, label="Observed", s=10)
plt.plot(x_vals, fitted_drift, color='red', label="Fitted drift", linewidth=2)
plt.xlabel("Log price at t")
plt.ylabel("Δ Log price")
plt.title(f"Drift Estimation for {ticker}")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(df.index[1:], np.exp(x_vals), label='Price')
plt.axhline(y=np.exp(mu), color='r', linestyle='--', label=f'Long-term mean: ${np.exp(mu):.2f}')
plt.title(f"{ticker} Price and OU Long-term Mean")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Statistical significance check
if p_value < 0.05:
    print("\n✓ Mean reversion is statistically significant (p < 0.05)")
else:
    print(f"\n⚠ Mean reversion may not be statistically significant (p = {p_value:.4f})")
