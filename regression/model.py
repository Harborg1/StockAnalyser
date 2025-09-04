import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

TICKER = 'SPY'
INTERVAL = '1h'
PERIOD = '730d' if INTERVAL == '1h' else 'max'
SHIFT_RANGE = range(1, 120)
MACD_FAST = 12
MACD_SLOW = 27
MACD_SPAN = 9
MFI_LENGTH = 14
MFI_OVERBOUGHT = 70
MFI_OVERSOLD = 30
RSI_LENGTH = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
BB_LEN = 20
DEVS = 2
CUTOFF = 0.5
TRAIN_SIZE = .7

LOOKBACK = 10000

MODEL =  "logit"

STRATEGY = ['BB', 'MACD_hist', 'RSI', 'MFI']
OPTIMAL_SHIFT = None

def get_data(ticker=TICKER, lookback=LOOKBACK, interval=INTERVAL, plot = False):

    df = yf.download(ticker, interval=interval, auto_adjust=True, period=PERIOD)
    df.rename(columns={'Date': 'Datetime'}, inplace=True)
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()

    for c in df.select_dtypes(include=[np.number]).columns:
        df[f'{c}_change'] = df[c].pct_change().shift(1) * 100

    # only return the subset of data you are interested in
    subset = df.iloc[-lookback:, :]
    if plot:
        plt.figure()
        plt.plot(subset['Close'])
        plt.title(f'Price Movements for {ticker} During Study')

    return subset.dropna()

def add_BB(df, devs=DEVS, bb_len=BB_LEN, plot = False):

    # can change to ema (use MACD video/code for reference)
    df['BB_SMA'] = df['Close'].shift(1).rolling(bb_len).mean()

    # get the standard deviation of the close prices for the period
    df['BB_STD'] = df['Close'].shift(1).rolling(bb_len).std()

    df['Upper_Band'] = df['BB_SMA'] + (devs * df['BB_STD'])
    df['Lower_Band'] = df['BB_SMA'] - (devs * df['BB_STD'])

    df['BB'] = (df['Upper_Band'] - df['Close']) / (df['Upper_Band'] - df['Lower_Band'])

    df = df.dropna()

    if plot:

        plt.figure()
        plt.plot(df['Close'], color='blue')
        plt.plot(df['Upper_Band'], color='orange')
        plt.plot(df['Lower_Band'], color='orange')
        plt.title(f'{TICKER} Bollinger Bands. Len: {BB_LEN}, Deviations: {DEVS}');

    return df

def add_RSI(df, length=RSI_LENGTH, overbought=RSI_OVERBOUGHT, oversold=RSI_OVERSOLD, plot=False):

    price_change = df['Close'].diff()

    # separate gains/losses
    gain = price_change.where(price_change > 0, 0)
    loss = -price_change.where(price_change < 0, 0)

    # average gain vs loss
    avg_gain = gain.shift(1).rolling(window=length).mean()
    avg_loss = loss.shift(1).rolling(window=length).mean()

    # calculate rsi
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi

    # plot the relative strength index

    if plot:
        plt.figure()
        plt.plot(df['RSI'])
        plt.axhline(overbought, color='red')
        plt.axhline(oversold, color='green')
        plt.title('Relative Strength Index')

    return df.dropna()
def add_MACD(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    span: int = MACD_SPAN,
    plot: bool = False
) -> pd.DataFrame:
    """
    Compute MACD features using ONLY information available up to t-1
    to avoid lookahead. Optionally plot the histogram.
    """

    out = df.copy()

    # Use prior close so features at time t are based on data up to t-1
    c = out['Close'].shift(1)

    # Classic TA behavior with adjust=False
    out[f'{fast}_ema'] = c.ewm(span=fast, adjust=False).mean()
    out[f'{slow}_ema'] = c.ewm(span=slow, adjust=False).mean()

    out['MACD'] = out[f'{fast}_ema'] - out[f'{slow}_ema']
    out['Signal'] = out['MACD'].ewm(span=span, adjust=False).mean()
    out['MACD_hist'] = out['MACD'] - out['Signal']

    # Drop rows that are not fully formed yet
    out = out.dropna(subset=[f'{fast}_ema', f'{slow}_ema', 'MACD', 'Signal', 'MACD_hist'])

    if plot:
        plt.figure()
        plt.bar(x=range(len(out)), height=out['MACD_hist'])
        plt.title(f'{fast}-{slow}-{span} MACD Histogram')
        plt.tight_layout()
        plt.close()

    return out

def add_MFI(df, length=MFI_LENGTH, overbought=MFI_OVERBOUGHT, oversold=MFI_OVERSOLD, plot = False):
    df = df.copy()

    # Step 1: Calculate typical price
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3

    # Step 2: Calculate raw money flow
    df['Raw_Money_Flow'] = df['Typical_Price'] * df['Volume']

    # Step 3: Classify positive/negative money flow
    df['Price_Change'] = df['Typical_Price'].diff()

    df['Pos_Flow'] = np.where(df['Price_Change'] > 0, df['Raw_Money_Flow'], 0)
    df['Neg_Flow'] = np.where(df['Price_Change'] < 0, df['Raw_Money_Flow'], 0)

    # Step 4: Money Flow Ratio and MFI
    pos_sum = df['Pos_Flow'].shift(1).rolling(window=length).sum()
    neg_sum = df['Neg_Flow'].shift(1).rolling(window=length).sum()
    mfr = pos_sum / neg_sum
    df['MFI'] = 100 - (100 / (1 + mfr))

    # Step 5: Plot

    if plot: 
        plt.figure()
        plt.plot(df['MFI'], label='MFI')
        plt.axhline(overbought, color='red', linestyle='--', label='Overbought')
        plt.axhline(oversold, color='green', linestyle='--', label='Oversold')
        plt.title('Money Flow Index')
        plt.legend()
        #plt.show()
        plt.close()

    return df.dropna()


def add_target(df, shift):
    df = df.copy()
    df[f'Close + {shift}'] = df['Close'].shift(-shift)
    # require at least +0.5% gain over shift
    df['Target'] = ((df[f'Close + {shift}'] - df['Close']) / df['Close'] > 0.005).astype(int)
    return df.dropna().reset_index(drop=True)


def add_roc_plot(y_true, y_scores, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.close()


def plot_prediction_distribution(y_pred_prob):
    plt.figure()
    plt.hist(y_pred_prob, bins=50, color='gray')
    plt.title('Distribution of Logistic Predictions')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.tight_layout()
    #plt.show()
    plt.close()


def train_val_test_split(df, train_size=0.6, val_size=0.2):
    total_len = len(df)
    train_end = int(total_len * train_size)
    val_end = int(total_len * (train_size + val_size))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def backtest_strategy(
    df,
    shift,
    prob_threshold,
    prob_col="Score",
    original_df=None,
    fee_bps_per_side=1.0,
    slippage_bps_per_side=0.5,

    initial_capital=10000.0,
    require_after_cost=True,
):
    """
    Backtest assuming finite capital. Each trade uses 100% of capital, then updates
    capital after exit (no overlapping trades). Compare to buy-and-hold with same capital.
    """

    if INTERVAL == "1h":
        df = df.copy().dropna(subset=[prob_col, "Close", "Datetime"])
    else:
        df = df.copy().dropna(subset=[prob_col, "Close", "Date"])

    # Generate signals
    df["Buy_Signal"] = df[prob_col] > prob_threshold
    df["Entry_Price"] = df["Close"]
    df["Exit_Price"] = df["Close"].shift(-shift)

    if INTERVAL == "1h":
        df["Exit_Time"] = df["Datetime"].shift(-shift)
    else:
        df = df.copy().dropna(subset=[prob_col, "Close", "Date"])

    # Round-trip transaction cost in pct
    rt_cost_pct = 2.0 * (fee_bps_per_side + slippage_bps_per_side) / 1e4

    # Track capital evolution
    capital = initial_capital
    equity_curve = [capital]
    trade_log = []

    i = 0
    while i < len(df) - shift:
        row = df.iloc[i]
        if row["Score"] > prob_threshold:

            if INTERVAL == "1h":
                entry_time = row["Datetime"]
                entry_price = row["Close"]
            else: 
                entry_time = row["Date"]
                entry_price = row["Close"]

            exit_row = df.iloc[i + shift]
            if INTERVAL == "1h":
                exit_time = exit_row["Datetime"]

            else: 
                 exit_time = exit_row["Date"]

            exit_price = exit_row["Close"]

            gross_ret = (exit_price - entry_price) / entry_price
            net_ret = gross_ret - rt_cost_pct if require_after_cost else gross_ret

            capital *= (1 + net_ret)
            equity_curve.append(capital)

            trade_log.append({
                "Entry_Time": str(entry_time),
                "Exit_Time": str(exit_time),
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "GrossRetPct": gross_ret,
                "NetRetPct": net_ret,
                "Capital": capital,
                prob_col: row[prob_col],
            })

            i += shift  # skip forward by holding period
        else:
            i += 1

    trade_log = pd.DataFrame(trade_log)

    # Stats
    total_trades = len(trade_log)
    win_rate = (trade_log["NetRetPct"] > 0).mean() if total_trades > 0 else 0.0
    final_capital = capital

    print("\nStrategy Backtest (finite capital):")
    print(f"Initial Capital: {initial_capital:.2f}")
    print(f"Final Capital:   {final_capital:.2f}")
    print(f"Total Trades:    {total_trades}")
    print(f"Win Rate:        {win_rate:.2%}")
    print(f"Net Return:      {((final_capital / initial_capital) - 1):.2%}")

    # Buy and hold benchmark
    if original_df is not None and "Close" in original_df:
        bh_final = initial_capital * (
            original_df["Close"].iloc[-1] / original_df["Close"].iloc[0]
        )
        print(f"Buy & Hold Final Capital: {bh_final:.2f}")
        print(f"Buy & Hold Net Return:   {((bh_final / initial_capital) - 1):.2%}")

    trade_log.to_csv("csv_files/trades_with_dates.csv", index=True)

    return trade_log, equity_curve
def find_best_shift_by_pnl(
    train: pd.DataFrame, 
    val: pd.DataFrame, 
    model: str, 
    prob_threshold: float = CUTOFF,
    fee_bps_per_side: float = 1.0, 
    slippage_bps_per_side: float = 0.5,
    min_ret_pct: float = 0.0,
    require_after_cost: bool = True
) -> pd.DataFrame | None:
  
    print(f"Exploring metrics over SHIFT range for {TICKER} on {INTERVAL} interval with model={model}\n")
    results = []

    for shift in SHIFT_RANGE:
        try:
            # Generate future-return target column
            train_with_target = add_target(train.copy(), shift)
            val_with_target   = add_target(val.copy(), shift)

            # Extract features and targets, drop invalid rows
            X_train = train_with_target[STRATEGY].replace([np.inf, -np.inf], np.nan).dropna()
            y_train = train_with_target.loc[X_train.index, 'Target']

            X_val   = val_with_target[STRATEGY].replace([np.inf, -np.inf], np.nan).dropna()
            y_val   = val_with_target.loc[X_val.index, 'Target']

            # Skip fold if not enough samples or labels
            if len(y_val) < 10 or y_val.nunique() < 2 or y_train.nunique() < 2:
                continue

            # Train the classifier
            if model == "xgb":
                clf = XGBClassifier(
                    eval_metric="logloss", random_state=42,
                    n_estimators=100, max_depth=3, learning_rate=0.1
                )
                clf.fit(X_train, y_train)
                y_val_prob = clf.predict_proba(X_val)[:, 1]

            elif model == "logit":
                clf = LogisticRegression(
                    penalty="l2", solver="lbfgs", max_iter=1000,
                    class_weight="balanced", random_state=42
                )
                clf.fit(X_train, y_train)
                y_val_prob = clf.predict_proba(X_val)[:, 1]

            else:
                raise ValueError("model must be either 'xgb' or 'logit'")

            # AUC for classification performance
            auc_score = roc_auc_score(y_val, y_val_prob)

            # Align predictions with validation set for trade simulation
            val_trading_df = val_with_target.loc[X_val.index].copy()
            val_trading_df["Predicted_Prob"] = y_val_prob
            val_trading_df["Buy_Signal"] = val_trading_df["Predicted_Prob"] > prob_threshold
            val_trading_df["Exit_Price"] = val_trading_df["Close"].shift(-shift)

            # Filter rows with buy signal and valid exit price
            trades = val_trading_df[val_trading_df["Buy_Signal"]].dropna(subset=["Exit_Price"]).copy()

            # Baseline metrics
            base_up_rate = float(y_val.mean())                       # Proportion of upward moves in val
            coverage = float(val_trading_df["Buy_Signal"].mean())   # How often buy signals occur

            # Gross & Net returns
            trades["GrossRetPct"] = (trades["Exit_Price"] - trades["Close"]) / trades["Close"]
            round_trip_cost = 2.0 * (fee_bps_per_side + slippage_bps_per_side) / 1e4
            trades["NetRetPct"] = trades["GrossRetPct"] - round_trip_cost

            selected_returns = trades["NetRetPct"] if require_after_cost else trades["GrossRetPct"]
            winning_trades = selected_returns >= min_ret_pct

            total_trades = len(trades)
            win_rate = float(winning_trades.mean())

            # Calculate PnL in price points (not percent)
            val_pnl_points = float(
                ((trades["Exit_Price"] - trades["Close"]) - trades["Close"] * round_trip_cost).sum()
            ) if require_after_cost else float((trades["Exit_Price"] - trades["Close"]).sum())

            results.append({
                "Shift": shift,
                "AUC": auc_score,
                "Trades": total_trades,
                "Coverage": coverage,
                "BaseUpRate": base_up_rate,
                "WinRate": win_rate,
                "ValPnL": val_pnl_points
            })

        except Exception as e:
            print(f"Shift {shift:2d}: Error - {str(e)}")

    if not results:
        print("Warning: No valid shifts found in the specified range!")
        return None

    return pd.DataFrame(results).sort_values(by="ValPnL", ascending=False).reset_index(drop=True)




def run_shift_range_backtest(train, val, test, shift_range=SHIFT_RANGE, features=STRATEGY, model=MODEL, cutoff=CUTOFF):
    shift_results = []

    for shift in shift_range:
        try:
            # Evaluate model on test set
            test_df, _,_ = evaluate_on_test(train, val, test, shift=shift, 
                                            features=features, cutoff=cutoff, model=model)

            # Backtest on test predictions
            _, equity_curve = backtest_strategy(test_df, shift=shift, prob_threshold=cutoff, original_df=test,
                                                initial_capital=10000, require_after_cost=True)

            final_cap = equity_curve[-1] if equity_curve else 10000
            shift_results.append({
                "Shift": shift,
                "FinalCapital": final_cap
            })

        except Exception as e:
            print(f"Error at shift={shift}: {e}")

    results_df = pd.DataFrame(shift_results).sort_values("Shift")
    
    # Plot final capital vs. shift
    plt.figure(figsize=(12, 6))
    plt.plot(results_df["Shift"], results_df["FinalCapital"], marker='o')
    plt.xlabel("Shift (Holding Period)")
    plt.ylabel("Final Capital ($)")
    plt.title(f"Backtest Performance on Test Set across SHIFT_RANGE ({model})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return results_df

    
def evaluate_on_test(train_df: pd.DataFrame, val_df: pd.DataFrame,
                     test_df: pd.DataFrame, shift: int,
                     features: list, cutoff=CUTOFF, model=MODEL):
    """
    Retrain on train+val using the chosen shift, then evaluate on test_df.
    Supports either 'xgb' or 'logit' models.
    """
    # Clean
    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()
    val_df = val_df.replace([np.inf, -np.inf], np.nan).dropna()
    test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna()

    # Combine and prepare training data
    combined = pd.concat([train_df, val_df], ignore_index=True)
    combined = add_target(combined.copy(), shift).dropna()
    X_comb = combined[features]
    y_comb = combined['Target']

    # Prepare test data
    test_t = add_target(test_df.copy(), shift).dropna()
    X_test = test_t[features]
    y_test = test_t['Target']

    # Train and predict
    if model == "xgb":
        final_model = XGBClassifier(eval_metric='logloss', random_state=42)
        final_model.fit(X_comb, y_comb)
        y_pred_prob = final_model.predict_proba(X_test)[:, 1]

    elif model == "logit":
        final_model = LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=2000,
        class_weight="balanced", random_state=42
    )
        final_model.fit(X_comb, y_comb)
        y_pred_prob = final_model.predict_proba(X_test)[:, 1]

    else:
        raise ValueError("model must be either 'xgb' or 'logit'")

    test_t['Score'] = y_pred_prob
    test_t['Prediction'] = (test_t['Score'] > cutoff).astype(int)

    return test_t, y_test, y_pred_prob


def add_indicators(df):
    df = add_MACD(df)
    df= add_MFI(df)
    df = add_BB(df)
    df=add_RSI(df)

    return df

if __name__ == '__main__':
    df = get_data()

    # Split once
    train, val, test = train_val_test_split(df)

    # Apply indicators to each split (prevents leakage)
    train = add_indicators(train)

    val=add_indicators(val)

    test = add_indicators(test)

     #Apply indicators to full dataset for final model & plotting
    df = add_indicators(df)

    results_df = find_best_shift_by_pnl(train, val,model = MODEL)

    OPTIMAL_SHIFT = int(results_df.iloc[0]['Shift'])
    print(f"\nOptimal SHIFT based on validation ValPnL: {OPTIMAL_SHIFT}")


    test_df, y_test, y_prob = evaluate_on_test(train, val, test, shift=OPTIMAL_SHIFT, features=STRATEGY,cutoff=CUTOFF,
                                               model=MODEL)
    plot_prediction_distribution(y_prob)
    add_roc_plot(y_test, y_prob, title=f'ROC Curve (Test Set, SHIFT={OPTIMAL_SHIFT})')

    # Backtest

    backtest_strategy(test_df, shift=OPTIMAL_SHIFT,prob_threshold=CUTOFF, original_df=test)

    # # Evaluate across all shifts on the test set
    # test_shift_backtest_results = run_shift_range_backtest(train, val, test)
