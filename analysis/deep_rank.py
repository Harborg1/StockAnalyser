import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

# --- 0. CUDA CONFIGURATION ---
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
END_DATE = "2021-11-27"
HOLDING_PERIOD = 2     
PORTFOLIO_SIZE = 3     
_SIZE = 3   # Ensemble size to reduce variance
NUM_FEATURES = 5       

# GA Parameters (Table 4)
GA_GENERATIONS = 4    
GA_POPULATION = 12
GA_SELECTION_RATE = 0.5 
GA_MUTATION_RATE = 0.15 

# --- 2. EXTENSION: FEATURE ENGINEERING ---
def calc_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def get_technical_features(price_data):
    print("Generating Technical Indicators (Numpy Optimized)...")
    returns = price_data.pct_change()
    excess_returns = returns.sub(returns.mean(axis=1), axis=0)
    
    feature_map = {}
    
    for t in TICKERS:
        if t not in price_data.columns: continue
        try:
            p = price_data[t]
            
            ma = p.rolling(window=20).mean()
            ma_dev = (p - ma) / ma
            
            ema = p.ewm(span=20, adjust=False).mean()
            ema_dev = (p - ema) / ema
            
            ema12 = p.ewm(span=12, adjust=False).mean()
            ema26 = p.ewm(span=26, adjust=False).mean()
            macd = (ema12 - ema26) / p
            
            rsi = calc_rsi(p) / 100.0
            
            # Create DataFrame then convert to FLOAT32 NUMPY ARRAY immediately
            df = pd.DataFrame({
                'ER': excess_returns[t],
                'MA': ma_dev,
                'EMA': ema_dev,
                'MACD': macd,
                'RSI': rsi
            }).fillna(0)
            
            feature_map[t] = df.values.astype(np.float32)
            
        except: continue
            
    return feature_map, excess_returns

# --- 3. DYNAMIC MODEL ARCHITECTURE ---
class DynamicRankNet(nn.Module):
    def __init__(self, input_size, layer_structure):
        super(DynamicRankNet, self).__init__()
        layers = []
        in_dim = input_size
        for hidden_units in layer_structure:
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = hidden_units 
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, x):
        return self.model(x)

def train_ranknet(model, X_tensor, y_tensor, epochs, batch_size):
    X_tensor, y_tensor = X_tensor.to(device), y_tensor.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    model.train()
    
    for _ in range(epochs):
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
                diff = s1.unsqueeze(1) - s2.unsqueeze(0)
                loss = torch.log(1 + torch.exp(-diff)).mean()
                loss.backward()
                optimizer.step()
    return model

# --- 4. DATA FETCHING ---
print("Fetching data...")
data = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)['Close']
data = data.ffill().bfill()
feature_dict, excess_returns = get_technical_features(data)

# --- 5. GENETIC OPTIMIZER WITH TABLE 4 CONSTRAINTS ---
def run_mini_backtest(params, valid_tickers, current_idx):
    # EXTRACT GENES
    window = int(params['number_of_lags'])
    train_history = int(params['number_of_periods'])
    layer_config = params['layers']
    bs = int(params['batch_size'])
    ep = int(params['stopping_criterion'])
    
    train_end = current_idx - HOLDING_PERIOD
    train_start = max(0, train_end - train_history) # Dynamic History
    
    X_train, y_train = [], []
    
    # BUILD TRAINING SET
    for t in range(train_start, train_end, HOLDING_PERIOD):
        f_perf = excess_returns.iloc[t : t + HOLDING_PERIOD].sum()
        med = f_perf.median()
        
        for s in valid_tickers:
            start_slice = t - window
            if start_slice < 0: continue
            
            # Numpy slicing
            feat_block = feature_dict[s][start_slice : t].flatten()
            
            if len(feat_block) == window * NUM_FEATURES:
                X_train.append(feat_block)
                y_train.append(1.0 if f_perf[s] > med else 0.0)
                
    if len(X_train) < 50: return -999.0
    
    X_t = torch.tensor(np.array(X_train), dtype=torch.float32)
    t_mean, t_std = X_t.mean(), X_t.std() + 1e-8
    X_t = (X_t - t_mean) / t_std
    y_t = torch.tensor(np.array(y_train), dtype=torch.float32).view(-1, 1)
    
    input_dim = window * NUM_FEATURES
    model = DynamicRankNet(input_dim, layer_config)
    
    # Train using the specific gene's batch size and epochs
    model = train_ranknet(model, X_t, y_t, epochs=ep, batch_size=bs)
    
    # INFERENCE
    X_live, live_s = [], []
    for s in valid_tickers:
        start_slice = current_idx - window
        if start_slice < 0: continue
        
        feat_block = feature_dict[s][start_slice : current_idx].flatten()
        if len(feat_block) == window * NUM_FEATURES:
            X_live.append(feat_block)
            live_s.append(s)
            
    if not X_live: return -999.0
    
    X_live_t = torch.tensor(np.array(X_live), dtype=torch.float32)
    X_live_t = (X_live_t - t_mean) / t_std
    X_live_t = X_live_t.to(device)
    
    model.eval()
    with torch.no_grad():
        preds = model(X_live_t).cpu().numpy().flatten()
        
    score_df = pd.DataFrame({'Ticker': live_s, 'Score': preds}).sort_values('Score', ascending=False)
    top_picks = score_df.head(PORTFOLIO_SIZE)['Ticker']
    realized_ret = excess_returns.iloc[current_idx : current_idx + HOLDING_PERIOD][top_picks].sum().mean()
    return realized_ret

def evolve_hyperparams(current_idx):
    print(f"\nRunning Static Genetic Optimization (Index {current_idx})...")
    
    population = []
    for _ in range(GA_POPULATION):
        # TABLE 4 CONSTRAINTS APPLIED HERE
        
        # "Number of hidden" (Layers): 1 to 16
        depth = random.randint(1, 3) 
        # "Neurons" (Units per layer): 22 to 70
        layers = [random.randint(22, 70) for _ in range(depth)]
        
        gene = {
            # "Number of lags": 1 to 16
            'number_of_lags': random.randint(1, 16), 
            
            # "Batch size": 12 to 50
            'batch_size': random.randint(12, 50),
            
            # "Stopping Criterion" (Epochs): 11 to 24
            'stopping_criterion': random.randint(11, 24),
            
            # "Number of periods" (Training History): 55 to 259
            'number_of_periods': random.randint(55, 259),
            
            'layers': layers
        }
        population.append(gene)

    valid_tickers = [t for t in TICKERS if t in feature_dict]

    for gen in range(GA_GENERATIONS):
        scored = []
        for p in population:
            fitness = run_mini_backtest(p, valid_tickers, current_idx - HOLDING_PERIOD)
            scored.append((fitness, p))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        print(f"  > Gen {gen+1} Best Score: {scored[0][0]:.4f}")
        
        parents = [p for s, p in scored[:int(GA_POPULATION * GA_SELECTION_RATE)]]
        
        next_gen = parents.copy()
        while len(next_gen) < GA_POPULATION:
            p1, p2 = random.sample(parents, 2)
            
            # CROSSOVER
            child = {
                'number_of_lags': random.choice([p1['number_of_lags'], p2['number_of_lags']]),
                'batch_size': random.choice([p1['batch_size'], p2['batch_size']]),
                'stopping_criterion': random.choice([p1['stopping_criterion'], p2['stopping_criterion']]),
                'number_of_periods': random.choice([p1['number_of_periods'], p2['number_of_periods']]),
                'layers': random.choice([p1['layers'], p2['layers']])
            }
            
            # MUTATION (Re-roll with Table 4 limits)
            if random.random() < GA_MUTATION_RATE:
                mutation_type = random.choice(['lags', 'batch', 'stop', 'periods'])
                if mutation_type == 'lags':
                    child['number_of_lags'] = random.randint(1, 16)
                elif mutation_type == 'batch':
                    child['batch_size'] = random.randint(12, 50)
                elif mutation_type == 'stop':
                    child['stopping_criterion'] = random.randint(11, 24)
                elif mutation_type == 'periods':
                    child['number_of_periods'] = random.randint(55, 259)
                    
            next_gen.append(child)
        population = next_gen
        
    return scored[0][1]

# --- 6. CORRECTED ROLLING BACKTEST (STATIC OPTIMIZATION) ---
print(f"\nStarting Rolling Backtest on {device}...")
returns_best, returns_worst, returns_market, dates = [], [], [], []
cap_best, cap_worst, cap_market = 10000.0, 10000.0, 10000.0
valid_days = excess_returns.index

# Start backtest after enough data for indicators
start_idx = 300 

# STEP 1: Run Genetic Algorithm ONCE to find the best architecture
print(f"--- Optimizing Hyperparameters (Running GA Once) ---")
best_gene = evolve_hyperparams(start_idx)
print(f"Optimal Parameters Found: {best_gene}")
print(f"----------------------------------------------------")

# Extract the winning parameters to use for the ENTIRE backtest
window = best_gene['number_of_lags']
layer_config = best_gene['layers']
bs = best_gene['batch_size']
ep = best_gene['stopping_criterion']
train_history = best_gene['number_of_periods']

# STEP 2: Run the Rolling Backtest using these fixed parameters
for i in range(start_idx, len(valid_days) - HOLDING_PERIOD, HOLDING_PERIOD):
    
    # 3. Build Training Set (Historical)
    train_end = i
    # Fix: Use dynamic training history size from GA
    train_start = max(0, train_end - train_history)
    
    X_train, y_train = [], []
    valid_tickers = [t for t in TICKERS if t in feature_dict]
    
    for t in range(train_start, train_end, HOLDING_PERIOD):
        f_perf = excess_returns.iloc[t : t + HOLDING_PERIOD].sum()
        med = f_perf.median()
        for s in valid_tickers:
            start_slice = max(0, t - window)
            
            # Numpy Slicing
            feat_block = feature_dict[s][start_slice : t].flatten()
            
            if len(feat_block) == window * NUM_FEATURES:
                X_train.append(feat_block)
                y_train.append(1.0 if f_perf[s] > med else 0.0)
                
    if len(X_train) < 50: continue

    X_train_np = np.array(X_train)
    t_mean, t_std = X_train_np.mean(), X_train_np.std() + 1e-8
    X_tensor = (torch.tensor(X_train_np, dtype=torch.float32) - t_mean) / t_std
    y_tensor = torch.tensor(np.array(y_train), dtype=torch.float32).view(-1, 1)

    # 4. Train Ensemble & Predict
    scores_accum = np.zeros(0)
    live_tickers = []
    
    # Prepare Live Data
    X_live = []
    for s in valid_tickers:
        start_slice = max(0, i - window)
        
        # Numpy Slicing
        feat_block = feature_dict[s][start_slice : i].flatten()
        
        if len(feat_block) == window * NUM_FEATURES:
            X_live.append(feat_block)
            live_tickers.append(s)
            
    if not X_live: continue
    
    X_live_tensor = torch.tensor(np.array(X_live), dtype=torch.float32)
    X_live_tensor = (X_live_tensor - t_mean) / t_std
    X_live_tensor = X_live_tensor.to(device)
    
    scores_accum = np.zeros(len(live_tickers))
    
    # Train Ensemble (_SIZE models to reduce variance of the random weight initialization)
    for _ in range(_SIZE):
        input_dim = window * NUM_FEATURES
        model = DynamicRankNet(input_dim, layer_config)
        # Fix: Use dynamic epochs from GA
        model = train_ranknet(model, X_tensor, y_tensor, epochs=ep, batch_size=bs)
        model.eval()
        with torch.no_grad():
            scores_accum += model(X_live_tensor).cpu().numpy().flatten()
            
    # 5. Trading
    score_df = pd.DataFrame({'Ticker': live_tickers, 'Score': scores_accum / _SIZE}).sort_values('Score', ascending=False)
    
    top = score_df.head(PORTFOLIO_SIZE)['Ticker']
    bottom = score_df.tail(PORTFOLIO_SIZE)['Ticker']
    
    p0 = data.iloc[i]
    p1 = data.iloc[i + HOLDING_PERIOD]
    
    ret_b = ((p1[top] - p0[top]) / p0[top]).mean()
    ret_w = ((p1[bottom] - p0[bottom]) / p0[bottom]).mean()
    ret_m = ((p1[live_tickers] - p0[live_tickers]) / p0[live_tickers]).mean()
    
    cap_best *= (1 + ret_b)
    cap_worst *= (1 + ret_w)
    cap_market *= (1 + ret_m)
    
    dates.append(valid_days[i + HOLDING_PERIOD])
    returns_best.append(cap_best)
    returns_worst.append(cap_worst)
    returns_market.append(cap_market)

    if i % 10 == 0:
        print(f"Date: {valid_days[i].date()} | Best: ${cap_best:,.0f} | Worst: ${cap_worst:,.0f}")

# --- 7. PLOT ---
plt.figure(figsize=(12, 6))
plt.plot(dates, returns_best, label='Deep RankNet (Best)', color='green', linewidth=2)
plt.plot(dates, returns_market, label='Market', color='gray', linestyle='--')
plt.plot(dates, returns_worst, label='Losers (Worst)', color='red', alpha=0.6)
plt.title(f'Deep RankNet Performance')
plt.ylabel('Portfolio Value ($)')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

