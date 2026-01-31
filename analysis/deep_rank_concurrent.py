
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import concurrent.futures


# Pre-determinism:
#Optimal Parameters Found: {'lags': 6, 'batch': 46, 'epochs': 16, 'history': 190, 'layers': [31, 45]}
#Optimal Parameters Found: {'lags': 9, 'batch': 46, 'epochs': 15, 'history': 156, 'layers': [64, 55, 57, 60]}
#Optimal Parameters Found: {'lags': 6, 'batch': 46, 'epochs': 23, 'history': 190, 'layers': [31, 45]}
#Optimal Parameters Found: {'lags': 16, 'batch': 17, 'epochs': 17, 'history': 157, 'layers': [48, 53]}


# Post Determinism
#Optimal Parameters Found: {'lags': 3, 'batch': 40, 'epochs': 23, 'history': 195, 'layers': [63, 51, 31, 38]}
#Date: 2017-01-18 | Best: $10,097 | Market: $9,991 | Worst: $10,028
#Date: 2017-02-15 | Best: $10,117 | Market: $10,364 | Worst: $10,093
#Date: 2017-03-16 | Best: $10,096 | Market: $10,565 | Worst: $10,340

#Optimal Parameters Found: {'lags': 3, 'batch': 40, 'epochs': 23, 'history': 195, 'layers': [63, 51, 31, 38]}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 2. Fortæller PyTorch at den skal fejle eller advare, hvis en ikke-deterministisk metode bruges
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"--- Seed set to {seed} ---") # Udkommenteret for ikke at spamme konsollen



# --- 0. GLOBAL CONFIGURATION ---
TICKERS = [
    "AAPL", "XOM", "MSFT", "GOOGL", "JNJ", "WFC", "GE", "WMT", "JPM", "CVX",
    "PG", "VZ", "PFE", "IBM", "KO", "CSCO", "ORCL", "DIS", "INTC", "MRK",
    "V", "PEP", "HD", "T", "SLB", "UNH", "AMZN", "META", "CMCSA",
    "MCD", "BA", "MMM", "BMY", "HON", "UNP", "AMGN", "C", "GILD", "ABBV",
    "MO", "NKE", "LLY", "ACN", "TXN", "AVGO", "COST", "MDT", "QCOM", "PYPL"
]

# Backtest Params
START_DATE = "2015-11-06"
END_DATE = "2021-11-27"
HOLDING_PERIOD = 2      
PORTFOLIO_SIZE = 3      
NUM_MODELS = 3          # Ensemble size
NUM_FEATURES = 5        

# Genetic Algorithm Params (Table 4)
GA_GENERATIONS = 4    
GA_POPULATION = 48    
GA_SELECTION_RATE = 0.5 
GA_MUTATION_RATE = 0.15 
RETRAIN_FREQ = 20

# --- 1. DATA HELPERS ---
def calc_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

# --- 2. ENGINE CLASS: Neural Network Logic ---
class DeepRankNetEngine:
    """
    The 'Workhorse': Handles model creation, training, and prediction.
    Defined at top-level so worker processes can find it.
    """
    def __init__(self, input_dim, layer_config, device):
        self.device = device
        self.input_dim = input_dim
        self.model = self._build_model(input_dim, layer_config).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.t_mean = 0
        self.t_std = 1

    def _build_model(self, input_size, layer_structure):
        layers = []
        in_dim = input_size
        for hidden_units in layer_structure:
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = hidden_units 
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def train(self, X_numpy, y_numpy, epochs, batch_size):
        self.model.train()
        X_t = torch.tensor(X_numpy, dtype=torch.float32)
        
        # Normalize and store stats for inference later
        self.t_mean, self.t_std = X_t.mean(), X_t.std() + 1e-8
        X_t = (X_t - self.t_mean) / self.t_std
        y_t = torch.tensor(y_numpy, dtype=torch.float32).view(-1, 1)
        
        X_t, y_t = X_t.to(self.device), y_t.to(self.device)
        
        for _ in range(epochs):
            permutation = torch.randperm(X_t.size()[0])
            for i in range(0, X_t.size()[0], batch_size):
                indices = permutation[i:i+batch_size]
                batch_X, batch_y = X_t[indices], y_t[indices]
                
                # Pairwise Loss Logic
                idx1 = (batch_y == 1).nonzero(as_tuple=True)[0]
                idx2 = (batch_y == 0).nonzero(as_tuple=True)[0]

                if len(idx1) > 0 and len(idx2) > 0:
                    self.optimizer.zero_grad()
                    scores = self.model(batch_X)
                    s1, s2 = scores[idx1], scores[idx2]
                    diff = s1.unsqueeze(1) - s2.unsqueeze(0)
                    loss = torch.log(1 + torch.exp(-diff)).mean()
                    loss.backward()
                    self.optimizer.step()

    def predict(self, X_numpy):
        self.model.eval()
        X_t = torch.tensor(X_numpy, dtype=torch.float32)
        X_t = (X_t - self.t_mean) / self.t_std
        X_t = X_t.to(self.device)
        with torch.no_grad():
            return self.model(X_t).cpu().numpy().flatten()

# --- 3. OPTIMIZER CLASS: Parallel Genetic Algorithm ---
class GeneticOptimizer:
    def __init__(self, feature_map, excess_returns, tickers, pop_size=GA_POPULATION, generations=GA_GENERATIONS):
        self.features = feature_map
        self.returns = excess_returns
        self.tickers = [t for t in tickers if t in feature_map]
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = GA_MUTATION_RATE
        self.selection_rate = GA_SELECTION_RATE

    # STATIC METHOD: Safe for Multiprocessing
    @staticmethod
    def _evaluate_gene_worker(gene, current_idx, features, returns, tickers,seed):
        # Unpack Gene
        set_seed(seed)
        torch.set_num_threads(1)
        window, history = gene['lags'], gene['history']
        train_end = current_idx - HOLDING_PERIOD
        train_start = max(0, train_end - history)
        
        X, y = [], []
        
        # Build Mini-Dataset
        for t in range(train_start, train_end, HOLDING_PERIOD):
            f_perf = returns.iloc[t : t + HOLDING_PERIOD].sum()
            med = f_perf.median()
            for s in tickers:
                s_slice = t - window
                if s_slice < 0: continue
                feat = features[s][s_slice:t].flatten()
                if len(feat) == window * NUM_FEATURES:
                    X.append(feat)
                    y.append(1.0 if f_perf[s] > med else 0.0)
                    
        if len(X) < 50: return -999.0
        
        # Force CPU for the worker process to avoid CUDA collisions
        local_device = torch.device("cpu")
        
        # Instantiate Engine (The Class we defined above)
        engine = DeepRankNetEngine(
            input_dim=window * NUM_FEATURES, 
            layer_config=gene['layers'], 
            device=local_device
        )
        
        engine.train(np.array(X), np.array(y), epochs=gene['epochs'], batch_size=gene['batch'])
        
        # Validation Inference
        X_live, live_s = [], []
        for s in tickers:
            s_slice = current_idx - window
            if s_slice < 0: continue
            feat = features[s][s_slice:current_idx].flatten()
            if len(feat) == window * NUM_FEATURES:
                X_live.append(feat); live_s.append(s)
                
        if not X_live: return -999.0
        preds = engine.predict(np.array(X_live))
        
        # Score based on next HOLDING_PERIOD return
        df = pd.DataFrame({'T': live_s, 'S': preds}).sort_values('S', ascending=False)
        top = df.head(PORTFOLIO_SIZE)['T']
        return returns.iloc[current_idx : current_idx + HOLDING_PERIOD][top].sum().mean()

    def _create_gene(self):
        # Table 4 Constraints
        depth = random.randint(1, 4)
        layers = [random.randint(22, 70) for _ in range(depth)]
        return {
            'lags': random.randint(1, 16),
            'batch': random.randint(12, 50),
            'epochs': random.randint(11, 24),
            'history': random.randint(55, 259),
            'layers': layers
        }

    def optimize(self, current_idx):
        set_seed(42) # Låser start-populationen
        print(f"  [GA] Evolving parameters for index {current_idx}...")
        
        population = [self._create_gene() for _ in range(self.pop_size)]
        
        for g in range(self.generations):
            # 1. Opret en tom liste med faste pladser
            results_by_index = [None] * self.pop_size 
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Map future -> index (så vi ved hvor resultatet hører hjemme)
                future_to_index = {}
                
                for i, gene in enumerate(population):
                    worker_seed = 42 + (current_idx * 10000) + (g * 1000) + i
                    fut = executor.submit(
                        GeneticOptimizer._evaluate_gene_worker, 
                        gene, 
                        current_idx, 
                        self.features, 
                        self.returns, 
                        self.tickers,
                        worker_seed
                    )
                    future_to_index[fut] = i

                # Saml resultaterne
                for fut in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[fut]
                    try:
                        raw_score = fut.result()
                        # Vi afrunder til 6 decimaler for at fjerne mikroskopisk "støj"
                        clean_score = round(raw_score, 6) 
                        results_by_index[idx] = (clean_score, population[idx])
                    except Exception:
                        results_by_index[idx] = (-999.0, population[idx])

            # Nu er rækkefølgen i 'results_by_index' altid identisk (0, 1, 2...)
            # Derfor vil sorteringen af 'ties' også være deterministisk.
            scored = sorted(results_by_index, key=lambda x: x[0], reverse=True)
            
            print(f"    > Gen {g+1} Top Score: {scored[0][0]:.4f}")
            
            # 2. Lås Evolutionen (Vigtigt!)
            set_seed(42 + g) 
            
            parents = [p for s, p in scored[:int(self.pop_size * self.selection_rate)]]
            next_gen = parents.copy()
            
            while len(next_gen) < self.pop_size:
                p1, p2 = random.sample(parents, 2)
                child = {k: random.choice([p1[k], p2[k]]) for k in p1}
                
                if random.random() < self.mutation_rate:
                    fresh = self._create_gene()
                    key = random.choice(list(child.keys()))
                    child[key] = fresh[key]
                next_gen.append(child)
            
            population = next_gen
            
        return scored[0][1]

# --- 4. MAIN EXECUTION BLOCK ---

    # --- 4. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Start med et clean seed
    set_seed(42)

    # 1. SETUP
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main Process Device: {DEVICE}")

    # --- ÆNDRING: DATA CACHING (LÅSER DATA FAST) ---
    data_file = "locked_data.csv"
    
    if os.path.exists(data_file):
        print("Loading data from local cache (Deterministic)...")
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    else:
        print("Downloading data from Yahoo (Network dependent)...")
        data = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)['Close']
        data = data.ffill().bfill()
        data.to_csv(data_file) # Gemmer data så de ALDRIG ændrer sig igen
        print(f"Data saved to {data_file}")

    # Sikr at vi kun bruger tickers der faktisk er i datafilen
    available_tickers = [t for t in TICKERS if t in data.columns]
    
    # Feature Engineering
    print("Generating Features...")
    returns = data.pct_change()
    excess_returns = returns.sub(returns.mean(axis=1), axis=0)
    feature_map = {}
    for t in available_tickers:
        if t not in data.columns: continue
        try:
            p = data[t]
            ma = p.rolling(20).mean(); ma_dev = (p-ma)/ma
            ema = p.ewm(span=20).mean(); ema_dev = (p-ema)/ema
            ema12 = p.ewm(span=12).mean(); ema26 = p.ewm(span=26).mean(); macd = (ema12-ema26)/p
            rsi = calc_rsi(p)/100.0
            
            df = pd.DataFrame({'ER': excess_returns[t], 'MA': ma_dev, 'EMA': ema_dev, 'MACD': macd, 'RSI': rsi}).fillna(0)
            feature_map[t] = df.values.astype(np.float32)
        except: continue

    valid_days = excess_returns.index
    start_idx = 300
    
    # 2. OPTIMIZATION PHASE
    # Instantiate the optimizer
    optimizer = GeneticOptimizer(feature_map, excess_returns, TICKERS)
    
    print("\n--- Phase 1: Running Genetic Optimization ---")
    # This runs the parallel GA and returns the single best configuration
    best_params = optimizer.optimize(start_idx)
    
    print(f"\nOptimal Parameters Found: {best_params}")
    
    # 3. BACKTEST PHASE (The Handoff)
    # ... (Nede i Phase 2) ...
    print("\n--- Phase 2: Running Rolling Backtest with Optimal Params ---")
    
    # Unpack the "Winning" Genes
    WIN_WINDOW = best_params['lags']
    WIN_LAYERS = best_params['layers']
    WIN_HISTORY = best_params['history']
    WIN_BATCH = best_params['batch']
    WIN_EPOCHS = best_params['epochs']
    
    # Benchmarking Variables
    dates = []
    returns_best, returns_worst, returns_market = [], [], []
    cap_best, cap_worst, cap_market = 10000.0, 10000.0, 10000.0
    
    # Cache til vores modeller
    current_ensemble = [] 
    
    # Tæller til at styre loopet
    loop_counter = 0

    for i in range(start_idx, len(valid_days) - HOLDING_PERIOD, HOLDING_PERIOD):
        
        # 1. Determinisme (Vigtigt for reproducibility)
        set_seed(42 + i)
        
        # 2. Data Preparation (Kører hver gang for at få X_live)
        # Vi skal stadig bruge X_train hvis vi skal træne, men vi kan optimere det væk hvis vi ikke skal.
        
        should_retrain = (loop_counter % (RETRAIN_FREQ // HOLDING_PERIOD) == 0)
        loop_counter += 1
        
        # Hent Live Data (Til prediction - skal bruges hver gang)
        X_live, live_s = [], []
        valid_tickers = [t for t in TICKERS if t in feature_map]
        
        for s in valid_tickers:
            s_slice = i - WIN_WINDOW
            if s_slice < 0: continue
            feat = feature_map[s][s_slice:i].flatten()
            if len(feat) == WIN_WINDOW * NUM_FEATURES:
                X_live.append(feat); live_s.append(s)
        
        if not X_live: continue
        X_live_np = np.array(X_live)

        # 3. Træning (KUN hvis det er tid til at gen-træne)
        if should_retrain or len(current_ensemble) == 0:
            #print(f"   [Retraining Models at index {i}]") # Debug print
            
            # Byg X_train og y_train nu
            train_end = i
            train_start = max(0, train_end - WIN_HISTORY)
            X_train, y_train = [], []
            
            for t in range(train_start, train_end, HOLDING_PERIOD):
                f_perf = excess_returns.iloc[t : t + HOLDING_PERIOD].sum()
                med = f_perf.median()
                for s in valid_tickers:
                    s_slice = t - WIN_WINDOW
                    if s_slice < 0: continue
                    feat = feature_map[s][s_slice:t].flatten()
                    if len(feat) == WIN_WINDOW * NUM_FEATURES:
                        X_train.append(feat)
                        y_train.append(1.0 if f_perf[s] > med else 0.0)
            
            if len(X_train) < 50: continue
            
            # Træn Ensemblet og gem det i cachen
            current_ensemble = []
            for _ in range(NUM_MODELS):
                engine = DeepRankNetEngine(
                    input_dim=WIN_WINDOW * NUM_FEATURES, 
                    layer_config=WIN_LAYERS, 
                    device=DEVICE
                )
                engine.train(
                    np.array(X_train), 
                    np.array(y_train), 
                    epochs=WIN_EPOCHS, 
                    batch_size=WIN_BATCH
                )
                current_ensemble.append(engine)
        
        # 4. Prediction (Bruger de cachede modeller)
        final_scores = np.zeros(len(live_s))
        
        for engine in current_ensemble:
            # Vigtigt: engine husker sin egen normalisering (mean/std) fra da den blev trænet
            final_scores += engine.predict(X_live_np)
            
        # D. Execute Trade & Benchmarking
        # ... (Resten af din kode er uændret herfra) ...
        score_df = pd.DataFrame({'Ticker': live_s, 'Score': final_scores}).sort_values('Score', ascending=False)
        
        top_picks = score_df.head(PORTFOLIO_SIZE)['Ticker']
        bottom_picks = score_df.tail(PORTFOLIO_SIZE)['Ticker']
        market_picks = live_s 
        
        p0 = data.iloc[i]
        p1 = data.iloc[i + HOLDING_PERIOD]
        
        ret_b = ((p1[top_picks] - p0[top_picks]) / p0[top_picks]).mean()
        ret_w = ((p1[bottom_picks] - p0[bottom_picks]) / p0[bottom_picks]).mean()
        ret_m = ((p1[market_picks] - p0[market_picks]) / p0[market_picks]).mean()
        
        cap_best *= (1 + ret_b)
        cap_worst *= (1 + ret_w)
        cap_market *= (1 + ret_m)
        
        dates.append(valid_days[i + HOLDING_PERIOD])
        returns_best.append(cap_best)
        returns_worst.append(cap_worst)
        returns_market.append(cap_market)
        
        if i % 20 == 0: # Print status lidt sjældnere
            print(f"Date: {valid_days[i].date()} | Best: ${cap_best:,.0f} | Market: ${cap_market:,.0f} | Worst: ${cap_worst:,.0f}")
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(dates, returns_best, label='Deep RankNet (Best)', color='green', linewidth=2)
    plt.plot(dates, returns_market, label='Market', color='gray', linestyle='--')
    plt.plot(dates, returns_worst, label='Losers (Worst)', color='red', alpha=0.6)
    plt.title('Final Backtest: Class-Based + Parallel GA')
    plt.ylabel('Portfolio Value ($)')
    plt.legend(); plt.grid(True, alpha=0.3); plt.show()
    