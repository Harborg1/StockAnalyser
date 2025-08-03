import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TICKER = 'SPY'

INTERVAL = '1h'

if INTERVAL == '1h':
    PERIOD = '730d'
else:

    PERIOD = 'max'

SHIFT_RANGE = range(1,120)
MACD_FAST = 12
MACD_SLOW = 27
MACD_SPAN = 9
MFI_LENGTH = 14
MFI_OVERBOUGHT = 70
MFI_OVERSOLD = 30
RSI_LENGTH = 14
RSI_OVERBOUGHT = 7
RSI_OVERSOLD = 30
BB_LEN = 20
DEVS = 2
LOOKBACK = 10000

STRATEGY = ['Volume', 'Open', 'High', 'Low', 'Close', 'MACD_hist', 'MFI', 'BB', 'RSI']

def get_data(ticker=TICKER, lookback=LOOKBACK, interval=INTERVAL):
    df = yf.download(ticker, interval=interval, auto_adjust=False, period=PERIOD)
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()

    df.rename(columns={'Date': 'Datetime'}, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    return df.iloc[-lookback:, :]

def add_MACD(df, fast=MACD_FAST, slow=MACD_SLOW, span=MACD_SPAN):
    df[f'{fast}_ema'] = df['Close'].ewm(span=fast).mean()
    df[f'{slow}_ema'] = df['Close'].ewm(span=slow).mean()
    df['MACD'] = df[f'{fast}_ema'] - df[f'{slow}_ema']
    df['Signal'] = df['MACD'].ewm(span=span).mean()
    df['MACD_hist'] = df['MACD'] - df['Signal']
    return df

def add_MFI(df, length=MFI_LENGTH, overbought=MFI_OVERBOUGHT, oversold=MFI_OVERSOLD):
    df = df.copy()
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Raw_Money_Flow'] = df['Typical_Price'] * df['Volume']
    df['Price_Change'] = df['Typical_Price'].diff()
    df['Pos_Flow'] = np.where(df['Price_Change'] > 0, df['Raw_Money_Flow'], 0)
    df['Neg_Flow'] = np.where(df['Price_Change'] < 0, df['Raw_Money_Flow'], 0)
    pos_sum = df['Pos_Flow'].rolling(window=length).sum()
    neg_sum = df['Neg_Flow'].rolling(window=length).sum()
    mfr = pos_sum / neg_sum
    df['MFI'] = 100 - (100 / (1 + mfr))
    return df.dropna()

def add_BB(df, devs=DEVS, bb_len=BB_LEN):
    df['BB_SMA'] = df['Close'].rolling(bb_len).mean()
    df['BB_STD'] = df['Close'].rolling(bb_len).std()
    df['Upper_Band'] = df['BB_SMA'] + (devs * df['BB_STD'])
    df['Lower_Band'] = df['BB_SMA'] - (devs * df['BB_STD'])
    df['BB'] = (df['Upper_Band'] - df['Close']) / (df['Upper_Band'] - df['Lower_Band'])
    return df.dropna()

def add_RSI(df, length=RSI_LENGTH, overbought=RSI_OVERBOUGHT, oversold=RSI_OVERSOLD):
    price_change = df['Close'].diff()
    gain = price_change.where(price_change > 0, 0)
    loss = -price_change.where(price_change < 0, 0)
    avg_gain = gain.rolling(window=length).mean()
    avg_loss = loss.rolling(window=length).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df.dropna()

def add_target(df, shift):
    if not isinstance(shift, int):
        raise ValueError(f"Expected integer shift, got {type(shift)}")
    df[f'Close + {shift}'] = df['Close'].shift(-shift)
    df['Target'] = (df[f'Close + {shift}'] > df['Close']).astype(int)
    return df


def generate_logistic_regression_output(df, features=STRATEGY, target='Target', splits=5):
    subset = df[features + [target, 'Datetime', 'Close']].dropna()
    dates = subset['Datetime']
    closes = subset['Close']
    X = subset[features]
    y = subset[target]


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=splits)
    all_results = []

    for _, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        close_test = closes.iloc[test_idx]
        date_test = dates.iloc[test_idx]

        model = LogisticRegression(penalty='l2', C=0.1, solver='liblinear', class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob > 0.6).astype(int)

        if y_test.nunique() < 2:
            continue

         # Ensure all arrays are converted to the same length before building the DataFrame
        date_array = np.array(date_test).reshape(-1)
        close_array = np.array(close_test).reshape(-1)
        target_array = np.array(y_test).reshape(-1)
        pred_array = np.array(y_pred).reshape(-1)
        prob_array = np.array(y_pred_prob).reshape(-1)

        min_len = min(len(date_array), len(close_array), len(target_array), len(pred_array), len(prob_array))

        df_out = pd.DataFrame({
            'Datetime': date_array[:min_len],
            'Close': close_array[:min_len],
            'Target': target_array[:min_len],
            'Prediction': pred_array[:min_len],
            'Probability': prob_array[:min_len]
        })

        all_results.append(df_out)

    test_df = pd.concat(all_results).reset_index(drop=True)
    print(test_df['Target'].value_counts(normalize=True))

    plt.figure()
    plt.hist(test_df['Probability'], bins=50)
    plt.title('Distribution of Logistic Predictions')
    plt.tight_layout()
    plt.show()

    return test_df, test_df['Target'], test_df['Probability']

def analyze_logistic_regresison(df, y_true, y_scores, title=f'{STRATEGY} ROC Plot'):
    cm = confusion_matrix(df['Target'], df['Prediction'])
    labels = ['Down (0)', 'Up (1)']

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure()
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='.0f')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df

def build_feature_set(df, shift):
    df = df.copy()

    # Step 1: Calculate indicators (only past data is used)
    df = add_MACD(df)
    df = add_MFI(df)
    df = add_BB(df)
    df = add_RSI(df)

    # Step 2: Add target AFTER indicators
    df = add_target(df, shift=shift)

    # Step 3: Drop any row that has NA in features or target
    feature_cols = STRATEGY + ['Target']
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df


def find_optimal_shift(df, shift_range):

    results = []
    for shift in shift_range:
        shift_val, auc_score = evaluate_shift(df, shift)
        #print(f"SHIFT={shift_val} -> AUC={auc_score:.4f}")
        results.append((shift_val, auc_score))

    results_df = pd.DataFrame(results, columns=['SHIFT', 'AUC'])
    best_shift = results_df.loc[results_df['AUC'].idxmax()]

    plt.figure()
    plt.plot(results_df['SHIFT'], results_df['AUC'], marker='o')
    plt.title("AUC vs SHIFT")
    plt.xlabel("SHIFT (Future Prediction Horizon)")
    plt.ylabel("AUC")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"Optimal SHIFT = {best_shift['SHIFT']} with AUC = {best_shift['AUC']:.4f}")
    return int(best_shift['SHIFT'])


def backtest_strategy(df, shift, threshold=0.6, original_df=None):
    df = df.copy()
    df = df.dropna(subset=['Prediction', 'Close'])

    df['Buy_Signal'] = df['Prediction'] > threshold
    df['Entry_Price'] = df['Close']
    df['Exit_Price'] = df['Close'].shift(-shift)
    df['PnL'] = np.where(df['Buy_Signal'], df['Exit_Price'] - df['Entry_Price'], 0)

    trades = df[df['Buy_Signal']].copy()
    trades = trades.dropna(subset=['Exit_Price'])

    if 'Datetime' in trades.columns:
        trade_log = trades[['Datetime', 'Entry_Price', 'Exit_Price', 'PnL', 'Prediction']].copy()
    else:
        trade_log = trades[['Entry_Price', 'Exit_Price', 'PnL', 'Prediction']].copy()
        trade_log['Datetime'] = pd.NaT

    trade_log['Datetime'] = trade_log['Datetime'].astype(str)
    trade_log.index.name = 'TradeIndex'

    total_trades = len(trade_log)
    winning_trades = (trade_log['PnL'] > 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_return = trade_log['PnL'].sum()

    print(f"\nStrategy Backtest:")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Total Return: {total_return:.2f}")

    if original_df is not None and 'Close' in original_df:
        buy_hold_return = original_df['Close'].iloc[-1] - original_df['Close'].iloc[0]
        print(f"Buy and Hold Return: {buy_hold_return:.2f} points")

    trade_log.to_csv("trades_with_dates.csv", index=True)

    return trade_log
def evaluate_shift(df_raw, shift, features=STRATEGY, target='Target', test_size=0.3):
    # Rebuild feature set for THIS shift (safe!)
    df_shifted = build_feature_set(df_raw.copy(), shift=shift)
    
    subset = df_shifted[features + [target]].dropna()
    if subset.empty:
        return shift, 0.0

    X = subset[features]
    y = subset[target]

    if y.nunique() < 2:
        print(f"Skipping SHIFT={shift}: only one class in target.")
        return shift, float("nan")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    try:
        model = sm.Logit(y_train, X_train_const).fit(disp=0)
        y_pred_prob = model.predict(X_test_const)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
    except Exception as e:
        print(f"Failed for SHIFT={shift}: {e}")
        roc_auc = 0.0

    return shift, roc_auc


def check_for_overfitting_with_shift_analysis(df_raw, train_ratio=0.8, shift_range=range(5, 150)):
    df_full = build_feature_set(df_raw.copy(), shift=optimal_shift)  # just to define the split point
    BUFFER = 300
    split_idx = int(len(df_full) * train_ratio)

    # Use raw data (not df_full) for building safe per-shift feature sets
    train_raw = df_raw.iloc[:split_idx]
    test_raw = df_raw.iloc[split_idx + BUFFER:]

    print(f"\nTrain period: {train_raw['Datetime'].iloc[0]} --> {train_raw['Datetime'].iloc[-1]}")
    print(f"Test period:  {test_raw['Datetime'].iloc[0]} --> {test_raw['Datetime'].iloc[-1]}")
    print(f"Train rows: {len(train_raw)}, Test rows: {len(test_raw)}")

    print("\nEvaluating shift on TRAINING SET")
    optimal_shift_train = find_optimal_shift(train_raw, shift_range)

    print("\nEvaluating shift on TEST SET")
    optimal_shift_test = find_optimal_shift(test_raw, shift_range)

    print(f"\nTrain Optimal SHIFT: {optimal_shift_train}, Test Optimal SHIFT: {optimal_shift_test}")


def main():
    original_df = get_data()
    
    # Step 1: Find best SHIFT
    global optimal_shift 
    optimal_shift = find_optimal_shift(original_df, SHIFT_RANGE)
    print(f"Using SHIFT = {optimal_shift}")


    # Step 2: Build feature set using the best shift
    df = build_feature_set(original_df, shift=optimal_shift)

    # Step 3: Train and test model
    test_df, _, _ = generate_logistic_regression_output(df)
    test_df = test_df.loc[:, ~test_df.columns.duplicated()]

    # Step 4: Backtest
    trades = backtest_strategy(test_df, shift=optimal_shift, threshold=0.5, original_df=original_df)

    check_for_overfitting_with_shift_analysis(original_df)

    return trades

df = main()


