import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

# --- 0. CUDA CONFIGURATION ---
# Checks for an NVIDIA GPU and sets the device accordingly [cite: 42, 43]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. CONFIGURATION ---
TICKERS = [
    "AAPL", "XOM", "MSFT", "GOOGL", "JNJ", "WFC", "GE", "WMT", "JPM", "CVX",
    "PG", "VZ", "PFE", "IBM", "KO", "CSCO", "ORCL", "DIS", "INTC", "MRK",
    "V", "PEP", "HD", "T", "SLB", "UNH", "AMZN", "META", "CMCSA",
    "MCD", "BA", "MMM", "BMY", "HON", "UNP", "AMGN", "C", "GILD", "ABBV",
    "MO", "NKE", "LLY", "ACN", "TXN", "AVGO", "COST", "MDT", "QCOM", "PYPL"
]

START_DATE = "2015-11-06"
END_DATE = "2019-11-27"
TRAIN_WINDOW = 20      # Input days [cite: 245]
HOLDING_PERIOD = 2     # Days to hold [cite: 414]
PORTFOLIO_SIZE = 3     # "k" parameter [cite: 457]
_SIZE = 5      # To reduce variance

# GA Parameters from Table 4 [cite: 410]
GA_GENERATIONS = 4     
GA_POPULATION = 12     
GA_SELECTION_RATE = 0.5 
GA_MUTATION_RATE = 0.15 

# --- 2. MODEL ARCHITECTURE ---
class RankNet(nn.Module):
    def __init__(self, input_size, neurons):
        super(RankNet, self).__init__()
        # Linear layers handle the non-linear market behaviors [cite: 41, 42]
        self.model = nn.Sequential(
            nn.Linear(input_size, neurons),
            nn.BatchNorm1d(neurons),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(neurons, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        ).to(device) # Move entire model to GPU 

    def forward(self, x):
        return self.model(x)

def train_ranknet(model, X_tensor, y_tensor, epochs=25, batch_size=32):
    # Ensure data is moved to the GPU before training 
    X_tensor, y_tensor = X_tensor.to(device), y_tensor.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    model.train()
    
    for epoch in range(epochs):
        permutation = torch.randperm(X_tensor.size()[0])
        for i in range(0, X_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_tensor[indices], y_tensor[indices]
            
            idx1 = (batch_y == 1).nonzero(as_tuple=True)[0]
            idx2 = (batch_y == 0).nonzero(as_tuple=True)[0]

            if len(idx1) > 0 and len(idx2) > 0:
                optimizer.zero_grad()
                scores = model(batch_X)
                s1, s2 = scores[idx1], scores[idx2]
                # Pairwise Logistic Loss: Learning relative ranking [cite: 216, 217]
                diff = s1.unsqueeze(1) - s2.unsqueeze(0)
                loss = torch.log(1 + torch.exp(-diff)).mean() #[cite: 217, 218]
                loss.backward()
                optimizer.step()
    return model

# --- 3. DATA PREPARATION ---
print("Fetching data...")
data = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)['Close']
data = data.ffill().bfill()
# Excess Return calculation relative to market index [cite: 225, 232]
excess_returns = data.pct_change().sub(data.pct_change().mean(axis=1), axis=0).dropna()

# --- 4. GENETIC OPTIMIZER WITH FUNCTIONAL FITNESS ---
def run_mini_backtest(params, ex_rets_slice):
    """Fitness Function evaluating return-to-risk [cite: 142, 420]"""
    hist_size = int(params['period_size'])
    neurons = int(params['neurons'])
    bs = int(params['batch_size'])
    
    test_idx = len(ex_rets_slice) - HOLDING_PERIOD - 1
    if test_idx - hist_size - TRAIN_WINDOW < 0: return -999.0
    
    X_list, y_list = [], []
    train_range = range(test_idx - hist_size, test_idx - HOLDING_PERIOD, HOLDING_PERIOD)
    for k in train_range:
        h_slice = ex_rets_slice.iloc[k - TRAIN_WINDOW : k]
        f_perf = ex_rets_slice.iloc[k : k + HOLDING_PERIOD].sum()
        med = f_perf.median() # Label based on median [cite: 591]
        for s in TICKERS:
            try:
                pat = h_slice[s].values
                if len(pat) == TRAIN_WINDOW and not np.isnan(pat).any():
                    X_list.append(pat); y_list.append(1.0 if f_perf[s] > med else 0.0)
            except: continue
    
    if len(X_list) < 20: return -999.0
    
    # Standardize inputs for stable training [cite: 27, 48]
    X_t = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_t = torch.tensor(np.array(y_list), dtype=torch.float32).view(-1, 1)
    X_t = (X_t - X_t.mean()) / (X_t.std() + 1e-8)
    
    model = RankNet(TRAIN_WINDOW, neurons)
    model = train_ranknet(model, X_t, y_t, epochs=15, batch_size=bs)
    
    live_pat = []
    for s in TICKERS: live_pat.append(ex_rets_slice.iloc[test_idx - TRAIN_WINDOW : test_idx][s].values)
    
    # Inference on GPU
    X_live = (torch.tensor(np.array(live_pat), dtype=torch.float32).to(device) - X_t.to(device).mean()) / (X_t.to(device).std() + 1e-8)
    
    model.eval()
    with torch.no_grad():
        preds = model(X_live).cpu().numpy().flatten()
    
    top_idx = np.argsort(preds)[-PORTFOLIO_SIZE:]
    real_perf = ex_rets_slice.iloc[test_idx : test_idx + HOLDING_PERIOD].sum().values[top_idx].mean()
    return real_perf

def evolve_hyperparams(ex_rets):
    """Genetic Algorithm for hyperparameter optimization [cite: 256, 388]"""
    print("\nStarting Genetic Optimization...")
    population = [{'period_size': random.randint(55, 259), 'neurons': random.randint(22, 70), 'batch_size': random.randint(12, 50)} 
                  for _ in range(GA_POPULATION)]

    tune_slice = ex_rets.iloc[:250] 

    for gen in range(GA_GENERATIONS):
        scored = []
        for p in population:
            fitness = run_mini_backtest(p, tune_slice)
            scored.append((fitness, p))
        
        scored.sort(key=lambda x: x[0], reverse=True) #[cite: 261]
        parents = [p for s, p in scored[:int(GA_POPULATION * GA_SELECTION_RATE)]] #[cite: 262, 267]
        
        print(f"Gen {gen+1} Best Fitness (Predicted Return): {scored[0][0]:.5f}")
        
        next_gen = parents.copy()
        while len(next_gen) < GA_POPULATION:
            p1, p2 = random.sample(parents, 2)
            # Crossover to combine traits [cite: 269, 273]
            child = {k: random.choice([p1[k], p2[k]]) for k in p1}
            # Mutation to prevent local minima traps [cite: 274, 275]
            if random.random() < GA_MUTATION_RATE:
                child['period_size'] = random.randint(55, 259)
            next_gen.append(child)
        population = next_gen
        
    return scored[0][1]

# --- 5. EXECUTION ---
best_params = evolve_hyperparams(excess_returns)
TRAINING_HISTORY = best_params['period_size']
OPT_NEURONS = best_params['neurons']
OPT_BATCH = best_params['batch_size']

print(f"\nOptimization Complete. Best Params: {best_params}")

# --- 6. ROLLING BACKTEST ---
print(f"\nStarting Rolling Backtest on {device}...")
returns_best, returns_worst, returns_market, dates = [], [], [], []
cap_best, cap_worst, cap_market = 10000.0, 10000.0, 10000.0
valid_days = excess_returns.index
backtest_start = max(250, TRAIN_WINDOW + TRAINING_HISTORY)

for i in range(backtest_start, len(valid_days) - HOLDING_PERIOD, HOLDING_PERIOD):
    # STEP A: BUILD TRAINING SET [cite: 255, 413]
    X_train_list, y_train_list = [], []
    history_range = range(i - TRAINING_HISTORY, i - HOLDING_PERIOD, HOLDING_PERIOD)
    
    for k in history_range:
        h_slice = excess_returns.iloc[k - TRAIN_WINDOW : k]
        f_perf = excess_returns.iloc[k : k + HOLDING_PERIOD].sum()
        median_f = f_perf.median()
        
        for stock in TICKERS:
            try:
                pattern = h_slice[stock].values
                if len(pattern) == TRAIN_WINDOW and not np.isnan(pattern).any():
                    X_train_list.append(pattern)
                    y_train_list.append(1.0 if f_perf[stock] > median_f else 0.0)
            except: continue
                
    if not X_train_list: continue

    X_train_np = np.array(X_train_list)
    t_mean, t_std = np.mean(X_train_np), np.std(X_train_np) + 1e-8
    X_tensor = (torch.tensor(X_train_np, dtype=torch.float32) - t_mean) / t_std
    y_tensor = torch.tensor(np.array(y_train_list), dtype=torch.float32).view(-1, 1)

    # STEP B:  PREDICTION
    current_slice = excess_returns.iloc[i - TRAIN_WINDOW : i]
    X_live_list, live_tickers = [], []
    for s in TICKERS:
        try:
            p = current_slice[s].values
            if len(p) == TRAIN_WINDOW and not np.isnan(p).any():
                X_live_list.append(p); live_tickers.append(s)
        except: continue
            
    # Move live data to GPU for inference [cite: 430]
    X_live_tensor = (torch.tensor(np.array(X_live_list), dtype=torch.float32).to(device) - t_mean) / t_std
    
    scores = np.zeros(len(live_tickers))
    for _ in range(_SIZE):
        model = RankNet(TRAIN_WINDOW, OPT_NEURONS)
        model = train_ranknet(model, X_tensor, y_tensor, epochs=25, batch_size=OPT_BATCH)
        model.eval()
        with torch.no_grad(): 
            # Bring scores back to CPU for aggregation
            scores += model(X_live_tensor).cpu().numpy().flatten()
    
    score_df = pd.DataFrame({'Ticker': live_tickers, 'Score': scores / _SIZE}).sort_values('Score', ascending=False)
    
    # STEP C: TRADING [cite: 416, 457]
    top, bottom = score_df.head(PORTFOLIO_SIZE)['Ticker'], score_df.tail(PORTFOLIO_SIZE)['Ticker']
    p0, p1 = data.iloc[i], data.iloc[i + HOLDING_PERIOD]
    
    ret_b = ((p1[top] - p0[top]) / p0[top]).mean()
    ret_w = ((p1[bottom] - p0[bottom]) / p0[bottom]).mean()
    ret_m = ((p1[live_tickers] - p0[live_tickers]) / p0[live_tickers]).mean()
    
    cap_best *= (1 + ret_b); cap_worst *= (1 + ret_w); cap_market *= (1 + ret_m)
    dates.append(valid_days[i + HOLDING_PERIOD])
    returns_best.append(cap_best); returns_worst.append(cap_worst); returns_market.append(cap_market)

    if i % 40 == 0:
        print(f"Date: {valid_days[i].date()} | Best: ${cap_best:,.0f} | Worst: ${cap_worst:,.0f}")

# --- 7. PLOT ---
plt.figure(figsize=(12, 6))
plt.plot(dates, returns_best, label='Optimized Winners (GA)', color='green', linewidth=2)
plt.plot(dates, returns_market, label='Market Benchmark', color='gray', linestyle='--')
plt.plot(dates, returns_worst, label='Optimized Losers', color='red', alpha=0.6)
plt.title('Deep RankNet with GPU Acceleration and GA Optimization')
plt.ylabel('Portfolio Value ($)')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()