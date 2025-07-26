import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

TICKER = 'SPY'
INTERVAL = '1h'

if INTERVAL == '1h':
    PERIOD = '730d'
else:
    PERIOD = 'max'

SHIFT = 3
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
    df = df.reset_index(drop=True)
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

def add_target(df, shift=SHIFT):
    df[f'Close + {shift}'] = df['Close'].shift(-shift)
    df['Target'] = (df[f'Close + {shift}'] > df['Close']) * 1
    return df

def generate_logistic_regression_output(df, features=STRATEGY, target='Target', test_size=0.2):
    subset = df[features + [target]].dropna()
    X = subset[features]
    y = subset[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    model = sm.Logit(y_train, X_train_const).fit()

    y_pred_prob = model.predict(X_test_const)

    plt.figure()
    plt.hist(y_pred_prob, bins=50)
    plt.title('Distribution of Logistic Predictions')
    plt.tight_layout()
    plt.show()

    y_pred = (y_pred_prob > 0.5).astype(int)
    print(model.summary())

    test_df = X_test.copy()
    test_df['Target'] = y_test
    test_df['Prediction'] = y_pred

    return test_df, y_test, y_pred_prob

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

def main():
    df = get_data()
    df = add_MACD(df)
    df = add_MFI(df)
    df = add_BB(df)
    df = add_RSI(df)
    df = add_target(df)
    test_df, y_test, y_pred_prob = generate_logistic_regression_output(df)
    test_df = analyze_logistic_regresison(test_df, y_true=y_test, y_scores=y_pred_prob)
    return test_df

df = main()
