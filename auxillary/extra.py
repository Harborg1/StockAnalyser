# import yfinance as yf
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, roc_curve, auc
# from sklearn.model_selection import train_test_split

# TICKER = 'SPY'
# INTERVAL = '1h'

# if INTERVAL == '1h':
#     PERIOD = '730d'
# else:

#     PERIOD = 'max'

# SHIFT = 9
# MACD_FAST = 12
# MACD_SLOW = 27
# MACD_SPAN = 9
# MFI_LENGTH = 14
# MFI_OVERBOUGHT = 70
# MFI_OVERSOLD = 30
# RSI_LENGTH = 14
# RSI_OVERBOUGHT = 7
# RSI_OVERSOLD = 30
# BB_LEN = 20
# DEVS = 2
# LOOKBACK = 10000

# STRATEGY = ['Volume', 'Open', 'High', 'Low', 'Close', 'MACD_hist', 'MFI', 'BB', 'RSI']

# def get_data(ticker=TICKER, lookback=LOOKBACK, interval=INTERVAL):
#     df = yf.download(ticker, interval=interval, auto_adjust=False, period=PERIOD)
#     df.columns = df.columns.get_level_values(0)
#     df = df.reset_index()
#     df = df.loc[:, ~df.columns.duplicated()]
#     return df.iloc[-lookback:, :]

# def add_MACD(df, fast=MACD_FAST, slow=MACD_SLOW, span=MACD_SPAN):
#     df[f'{fast}_ema'] = df['Close'].ewm(span=fast).mean()
#     df[f'{slow}_ema'] = df['Close'].ewm(span=slow).mean()
#     df['MACD'] = df[f'{fast}_ema'] - df[f'{slow}_ema']
#     df['Signal'] = df['MACD'].ewm(span=span).mean()
#     df['MACD_hist'] = df['MACD'] - df['Signal']
#     return df

# def add_MFI(df, length=MFI_LENGTH, overbought=MFI_OVERBOUGHT, oversold=MFI_OVERSOLD):
#     df = df.copy()
#     df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
#     df['Raw_Money_Flow'] = df['Typical_Price'] * df['Volume']
#     df['Price_Change'] = df['Typical_Price'].diff()
#     df['Pos_Flow'] = np.where(df['Price_Change'] > 0, df['Raw_Money_Flow'], 0)
#     df['Neg_Flow'] = np.where(df['Price_Change'] < 0, df['Raw_Money_Flow'], 0)
#     pos_sum = df['Pos_Flow'].rolling(window=length).sum()
#     neg_sum = df['Neg_Flow'].rolling(window=length).sum()
#     mfr = pos_sum / neg_sum
#     df['MFI'] = 100 - (100 / (1 + mfr))
#     return df.dropna()

# def add_BB(df, devs=DEVS, bb_len=BB_LEN):
#     df['BB_SMA'] = df['Close'].rolling(bb_len).mean()
#     df['BB_STD'] = df['Close'].rolling(bb_len).std()
#     df['Upper_Band'] = df['BB_SMA'] + (devs * df['BB_STD'])
#     df['Lower_Band'] = df['BB_SMA'] - (devs * df['BB_STD'])
#     df['BB'] = (df['Upper_Band'] - df['Close']) / (df['Upper_Band'] - df['Lower_Band'])
#     return df.dropna()

# def add_RSI(df, length=RSI_LENGTH, overbought=RSI_OVERBOUGHT, oversold=RSI_OVERSOLD):
#     price_change = df['Close'].diff()
#     gain = price_change.where(price_change > 0, 0)
#     loss = -price_change.where(price_change < 0, 0)
#     avg_gain = gain.rolling(window=length).mean()
#     avg_loss = loss.rolling(window=length).mean()
#     rs = avg_gain / avg_loss
#     df['RSI'] = 100 - (100 / (1 + rs))
#     return df.dropna()

# def add_target(df, shift=SHIFT):
#     df[f'Close + {shift}'] = df['Close'].shift(-shift)
#     df['Target'] = (df[f'Close + {shift}'] > df['Close']) * 1
#     return df

# def generate_logistic_regression_output(df, features=STRATEGY, target='Target', test_size=0.2):
#     subset = df[features + [target, 'Datetime', 'Close']].dropna()
    
#     dates = subset['Datetime']
#     closes = subset['Close']
    
#     X = subset[features]
#     y = subset[target]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, shuffle=False
#     )

#     X_train_const = sm.add_constant(X_train)
#     X_test_const = sm.add_constant(X_test)

#     model = sm.Logit(y_train, X_train_const).fit()
#     y_pred_prob = model.predict(X_test_const)
#     y_pred = (y_pred_prob > 0.5).astype(int)

#     print(model.summary())

#     test_df = X_test.copy()
#     test_df['Target'] = y_test
#     test_df['Prediction'] = y_pred
#     test_df['Datetime'] = dates.loc[X_test.index].values  # <-- fixed here
#     test_df['Close'] = closes.loc[X_test.index].values

#     plt.figure()
#     plt.hist(y_pred_prob, bins=50)
#     plt.title('Distribution of Logistic Predictions')
#     plt.tight_layout()
#     plt.show()

#     return test_df, y_test, y_pred_prob



# def analyze_logistic_regresison(df, y_true, y_scores, title=f'{STRATEGY} ROC Plot'):
#     cm = confusion_matrix(df['Target'], df['Prediction'])
#     labels = ['Down (0)', 'Up (1)']

#     cm_df = pd.DataFrame(cm, index=labels, columns=labels)
#     plt.figure()
#     sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='.0f')
#     plt.title("Confusion Matrix")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.tight_layout()
#     plt.show()

#     fpr, tpr, thresholds = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)

#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(title)
#     plt.legend(loc="lower right")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     return df


# def find_optimal_shift(df, shift_range):
#     df = add_MACD(df)
#     df = add_MFI(df)
#     df = add_BB(df)
#     df = add_RSI(df)

#     results = []
#     for shift in shift_range:
#         shift_val, auc_score = evaluate_shift(df, shift)
#         print(f"SHIFT={shift_val} -> AUC={auc_score:.4f}")
#         results.append((shift_val, auc_score))

#     results_df = pd.DataFrame(results, columns=['SHIFT', 'AUC'])
#     best_shift = results_df.loc[results_df['AUC'].idxmax()]

#     plt.figure()
#     plt.plot(results_df['SHIFT'], results_df['AUC'], marker='o')
#     plt.title("AUC vs SHIFT")
#     plt.xlabel("SHIFT (Future Prediction Horizon)")
#     plt.ylabel("AUC")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     print(f"Optimal SHIFT = {best_shift['SHIFT']} with AUC = {best_shift['AUC']:.4f}")
#     return int(best_shift['SHIFT'])

# def backtest_strategy(df, shift, threshold=0.6, original_df=None):
#     df = df.copy()
#     df = df.dropna(subset=['Prediction', 'Close'])

#     df['Buy_Signal'] = df['Prediction'] > threshold
#     df['Entry_Price'] = df['Close']
#     df['Exit_Price'] = df['Close'].shift(-shift)
#     df['PnL'] = np.where(df['Buy_Signal'], df['Exit_Price'] - df['Entry_Price'], 0)

#     trades = df[df['Buy_Signal']].copy()
#     trades = trades.dropna(subset=['Exit_Price'])

#     if 'Datetime' in trades.columns:
#         trade_log = trades[['Datetime', 'Entry_Price', 'Exit_Price', 'PnL', 'Prediction']].copy()
#     else:
#         trade_log = trades[['Entry_Price', 'Exit_Price', 'PnL', 'Prediction']].copy()
#         trade_log['Datetime'] = pd.NaT

#     trade_log['Datetime'] = trade_log['Datetime'].astype(str)
#     trade_log.index.name = 'TradeIndex'

#     total_trades = len(trade_log)
#     winning_trades = (trade_log['PnL'] > 0).sum()
#     win_rate = winning_trades / total_trades if total_trades > 0 else 0
#     total_return = trade_log['PnL'].sum()

#     print(f"\nStrategy Backtest:")
#     print(f"Total Trades: {total_trades}")
#     print(f"Win Rate: {win_rate:.2%}")
#     print(f"Total Return: {total_return:.2f}")

#     if original_df is not None and 'Close' in original_df:
#         buy_hold_return = original_df['Close'].iloc[-1] - original_df['Close'].iloc[0]
#         print(f"Buy and Hold Return: {buy_hold_return:.2f} points")

#     trade_log.to_csv("trades_with_dates.csv", index=True)

#     return trade_log




# def evaluate_shift(df, shift, features=STRATEGY, target='Target', test_size=0.2):
#     df_shifted = add_target(df.copy(), shift=shift)
#     subset = df_shifted[features + [target]].dropna()
    
#     if subset.empty:
#         return shift, 0.0  # Return zero AUC if not enough data
    
#     X = subset[features]
#     y = subset[target]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
#     X_train_const = sm.add_constant(X_train)
#     X_test_const = sm.add_constant(X_test)

#     try:
#         model = sm.Logit(y_train, X_train_const).fit(disp=0)
#         y_pred_prob = model.predict(X_test_const)
#         fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
#         roc_auc = auc(fpr, tpr)
#     except Exception as e:
#         print(f"Failed for SHIFT={shift}: {e}")
#         roc_auc = 0.0

#     return shift, roc_auc

# def main():
#     original_df = get_data()  # Save before adding indicators
#     df = original_df.copy()
    
#     optimal_shift = find_optimal_shift(df, shift_range=range(1, 150))

#     optimal_shift = 60
#     df = add_MACD(df)
#     df = add_MFI(df)
#     df = add_BB(df)
#     df = add_RSI(df)
#     df = add_target(df, shift=60)
#     test_df, y_test, y_pred_prob = generate_logistic_regression_output(df)
#     test_df = analyze_logistic_regresison(test_df, y_true=y_test, y_scores=y_pred_prob)
#     test_df = test_df.loc[:, ~test_df.columns.duplicated()]
#     trades = backtest_strategy(test_df, shift=optimal_shift, threshold=0.6, original_df=original_df)
#     return trades

# df = main() 
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import TimeSeriesSplit

# -------------------- Config --------------------
TICKER = 'SPY'
INTERVAL = '1h'
PERIOD = '730d' if INTERVAL == '1h' else 'max'

TEST_SIZE = 0.2        # fraction of data for final test
SHIFT_RANGE = range(1, 150)
CV_SPLITS = 10         # number of rolling folds
THRESHOLD = 0.6        # probability threshold for backtest

# Indicator parameters
MACD_FAST, MACD_SLOW, MACD_SPAN = 12, 27, 9
MFI_LENGTH = 14
BB_LEN, DEVS = 20, 2
RSI_LENGTH = 14
LOOKBACK = 10000

# Model pipeline: scale + logistic regression with L2
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000))
])

# -------------------- Feature & Label Engineering --------------------
def get_data(ticker=TICKER, lookback=LOOKBACK, interval=INTERVAL):
    df = yf.download(ticker, interval=interval, auto_adjust=False, period=PERIOD)
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df.loc[:, ~df.columns.duplicated()]
    return df.iloc[-lookback:, :]


def add_indicators(df):
    df = df.copy()
    # MACD histogram
    ema_fast = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=MACD_SPAN, adjust=False).mean()
    df['MACD_hist'] = macd - signal

    # Money Flow Index
    tp = df[['High','Low','Close']].mean(axis=1)
    rmf = tp * df['Volume']
    d_tp = tp.diff()
    pos_flow = np.where(d_tp > 0, rmf, 0)
    neg_flow = np.where(d_tp < 0, rmf, 0)
    pos_sum = pd.Series(pos_flow, index=df.index).rolling(window=MFI_LENGTH).sum()
    neg_sum = pd.Series(neg_flow, index=df.index).rolling(window=MFI_LENGTH).sum()
    df['MFI'] = 100 - (100 / (1 + pos_sum / neg_sum))

    # Bollinger Bands
    sma = df['Close'].rolling(window=BB_LEN).mean()
    std = df['Close'].rolling(window=BB_LEN).std()
    upper = sma + DEVS * std
    lower = sma - DEVS * std
    df['BB'] = (upper - df['Close']) / (upper - lower)

    # Relative Strength Index
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=RSI_LENGTH).mean()
    avg_loss = loss.rolling(window=RSI_LENGTH).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df.dropna()


def make_label(df, shift):
    d = df.copy()
    d['Target'] = (d['Close'].shift(-shift) > d['Close']).astype(int)
    return d.dropna()

# -------------------- Train/Test Split --------------------
def train_test_split_time(df, test_size=TEST_SIZE):
    split = int(len(df) * (1 - test_size))
    return df.iloc[:split], df.iloc[split:]

# -------------------- Rolling CV for Shift Selection --------------------

def select_optimal_shift(df, shift_range, cv_splits=CV_SPLITS):
    results = []
    for shift in shift_range:
        d = make_label(df, shift)
        X = d[['Volume','Open','High','Low','Close','MACD_hist','MFI','BB','RSI']]
        y = d['Target']
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        aucs = []
        for train_idx, val_idx in tscv.split(X):
            # Embargo to avoid leakage: drop last 'shift' rows from train slice
            valid_train_idx = train_idx[train_idx < len(X) - shift]
            X_tr, y_tr = X.iloc[valid_train_idx], y.iloc[valid_train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            pipeline.fit(X_tr, y_tr)
            prob = pipeline.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, prob)
            aucs.append(auc(fpr, tpr))
        results.append((shift, np.mean(aucs), np.std(aucs)))
    res = pd.DataFrame(results, columns=['SHIFT', 'AUC_MEAN', 'AUC_STD'])
    # 1-sigma rule: candidate shifts with mean >= max_mean - std
    max_mean = res['AUC_MEAN'].max()
    candidates = res[res['AUC_MEAN'] >= max_mean - res['AUC_STD']]
        # Instead of the one-sigma rule:
    best_shift = int(res.loc[res['AUC_MEAN'].idxmax(), 'SHIFT'])

    plt.figure()
    plt.errorbar(res['SHIFT'], res['AUC_MEAN'], yerr=res['AUC_STD'], fmt='o')
    plt.title('Shift Selection: Mean ± Std AUC')
    plt.xlabel('SHIFT')
    plt.ylabel('AUC')
    plt.tight_layout()
    plt.show()
    return best_shift, res

# -------------------- Final Fit & Backtest --------------------

def final_train_test(df, shift, test_size=TEST_SIZE):
    d = make_label(df, shift)
    train_df, test_df = train_test_split_time(d, test_size)
    # Embargo the last 'shift' rows of training
    train_df = train_df.iloc[:-shift]
    X_tr = train_df[['Volume','Open','High','Low','Close','MACD_hist','MFI','BB','RSI']]
    y_tr = train_df['Target']
    X_te = test_df[['Volume','Open','High','Low','Close','MACD_hist','MFI','BB','RSI']]
    y_te = test_df['Target']
    pipeline.fit(X_tr, y_tr)
    prob = pipeline.predict_proba(X_te)[:, 1]
    pred = (prob > 0.5).astype(int)
    result = test_df.copy()
    result['PredProb'] = prob
    result['Prediction'] = pred
    return result


def backtest(df, shift, threshold=THRESHOLD):
    d = df.copy().dropna(subset=['PredProb', 'Close'])
    d['Signal'] = d['PredProb'] > threshold
    d['Exit'] = d['Close'].shift(-shift)
    d['PnL'] = np.where(d['Signal'], d['Exit'] - d['Close'], 0)
    trades = d[d['Signal']].dropna(subset=['Exit'])
    print(f"Trades: {len(trades)}, WinRate: {trades['PnL'].gt(0).mean():.2%}, Total PnL: {trades['PnL'].sum():.2f}")
    return trades

# -------------------- Main --------------------
def main():
    raw = get_data()
    features = add_indicators(raw)
    best_shift, df_shifts = select_optimal_shift(features, SHIFT_RANGE)
    print(f"Selected SHIFT: {best_shift}")
    test_results = final_train_test(features, best_shift)
    fpr, tpr, _ = roc_curve(test_results['Target'], test_results['PredProb'])
    print(f"Test AUC: {auc(fpr, tpr):.4f}")
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--')
    plt.title('Test ROC')
    plt.tight_layout()
    plt.show()
    trades = backtest(test_results, best_shift)
    return best_shift, trades

if __name__ == "__main__":
    main() 