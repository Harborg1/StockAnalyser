import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from xgboost import XGBClassifier
TICKER = 'SPY'
INTERVAL = '1d'
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
CUTOFF = .5
TRAIN_SIZE = .7

LOOKBACK = 10000

STRATEGY = ['BB', 'MACD_hist', 'RSI', 'MFI']
OPTIMAL_SHIFT = None

def get_data(ticker=TICKER, lookback=LOOKBACK, interval=INTERVAL):
    df = yf.download(ticker, interval=interval, auto_adjust=True, period=PERIOD)
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()


    for c in df.select_dtypes(include=[np.number]).columns:
        df[f'{c}_change'] = df[c].pct_change().shift(1) * 100

    # only return the subset of data you are interested in
    subset = df.iloc[-lookback:, :]
    plt.figure()
    plt.plot(subset['Close'])
    plt.title(f'Price Movements for {ticker} During Study')

    return subset.dropna()

def add_BB(df, devs=DEVS, bb_len=BB_LEN):

    # can change to ema (use MACD video/code for reference)
    df['BB_SMA'] = df['Close'].shift(1).rolling(bb_len).mean()

    # get the standard deviation of the close prices for the period
    df['BB_STD'] = df['Close'].shift(1).rolling(bb_len).std()

    df['Upper_Band'] = df['BB_SMA'] + (devs * df['BB_STD'])
    df['Lower_Band'] = df['BB_SMA'] - (devs * df['BB_STD'])

    df['BB'] = (df['Upper_Band'] - df['Close']) / (df['Upper_Band'] - df['Lower_Band'])

    df = df.dropna()

    plt.figure()
    plt.plot(df['Close'], color='blue')
    plt.plot(df['Upper_Band'], color='orange')
    plt.plot(df['Lower_Band'], color='orange')
    plt.title(f'{TICKER} Bollinger Bands. Len: {BB_LEN}, Deviations: {DEVS}');

    return df

def add_RSI(df, length=RSI_LENGTH, overbought=RSI_OVERBOUGHT, oversold=RSI_OVERSOLD):

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
    plt.figure()
    plt.plot(df['RSI'])
    plt.axhline(overbought, color='red')
    plt.axhline(oversold, color='green')
    plt.title('Relative Strength Index')

    return df.dropna()

def add_MACD(df, fast=MACD_FAST, slow=MACD_SLOW, span=MACD_SPAN):

    df[f'{fast}_ema'] = df['Close'].ewm(span=fast).mean()
    df[f'{slow}_ema'] = df['Close'].ewm(span=slow).mean()

    # macd line is the difference betweent he fast and slow
    df[f'MACD'] = df[f'{fast}_ema'] - df[f'{slow}_ema']

    # macd signal is a 9-period moving average of this line
    df['Signal'] = df['MACD'].ewm(span=span).mean()

    # MACD histogram is almost always what is used in TA
    df['MACD_hist'] = df['MACD'] - df['Signal']

    # plot the histogram
    plt.figure()
    plt.bar(x=range(len(df)), height=df['MACD_hist'])
    plt.title(f'{MACD_FAST} - {MACD_SLOW} - {MACD_SPAN} MACD Histogram')

    return df

def add_MFI(df, length=MFI_LENGTH, overbought=MFI_OVERBOUGHT, oversold=MFI_OVERSOLD):
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
    pos_sum = df['Pos_Flow'].rolling(window=length).sum()
    neg_sum = df['Neg_Flow'].rolling(window=length).sum()
    mfr = pos_sum / neg_sum
    df['MFI'] = 100 - (100 / (1 + mfr))

    # Step 5: Plot
    plt.figure()
    plt.plot(df['MFI'], label='MFI')
    plt.axhline(overbought, color='red', linestyle='--', label='Overbought')
    plt.axhline(oversold, color='green', linestyle='--', label='Oversold')
    plt.title('Money Flow Index')
    plt.legend()
    #plt.show()

    return df.dropna()


def add_target(df, shift):
    df = df.copy()
    df[f'Close + {shift}'] = df['Close'].shift(-shift)
    threshold = 0.003  # e.g., 0.3% change
    df['Target'] = ((df[f'Close + {shift}'] - df['Close']) / df['Close'] > threshold).astype(int)
    return df.dropna().reset_index(drop=False)

def generate_regression_output(df, features=STRATEGY, target='Target', cutoff=CUTOFF):
    subset = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(subset) < 10:
        raise ValueError("Too few rows after cleaning.")

    X = sm.add_constant(subset[features])
    y = subset[target]

    model = sm.Logit(y, X).fit(disp=0)
    y_pred_prob = model.predict(X)

    df = df.loc[subset.index]
    df['Prediction'] = (y_pred_prob > cutoff).astype(int)
    return df, y, y_pred_prob


def generate_xgb_output(df, features=STRATEGY, target='Target', cutoff=CUTOFF):
    subset = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(subset) < 10:
        raise ValueError("Too few rows after cleaning.")
    
    X = subset[features]
    y = subset[target]

    model = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1
    )
    model.fit(X, y)
    y_pred_prob = model.predict_proba(X)[:, 1]

    df = df.loc[subset.index]
    df['Prediction'] = (y_pred_prob > cutoff).astype(int)
    return df, y, y_pred_prob

def add_confusion_matrix(df):
    cm = confusion_matrix(df['Target'], df['Prediction'])
    labels = ['Down (0)', 'Up (1)']
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure()
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='.0f')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    #plt.show()

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

def plot_prediction_distribution(y_pred_prob):
    plt.figure()
    plt.hist(y_pred_prob, bins=50, color='gray')
    plt.title('Distribution of Logistic Predictions')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.tight_layout()
    #plt.show()

def train_val_test_split(df, train_size=0.6, val_size=0.2):
    total_len = len(df)
    train_end = int(total_len * train_size)
    val_end = int(total_len * (train_size + val_size))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def train_and_test_model(train_df, test_df, shift, cutoff=CUTOFF):
    # Add target to both
    train_df = add_target(train_df.copy(), shift)
    test_df = add_target(test_df.copy(), shift)

    # Clean
    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()
    test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna()

    X_train = train_df[STRATEGY]
    y_train = train_df['Target']
    X_test = test_df[STRATEGY]
    y_test = test_df['Target']

    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    test_df = test_df.loc[X_test.index]
    test_df['Prediction'] = (y_pred_prob > cutoff).astype(int)

    return test_df, y_test, y_pred_prob


def backtest_strategy(df, shift, threshold=0.6, original_df=None):
    df = df.copy()
    df = df.dropna(subset=['Prediction', 'Close'])
    df['Buy_Signal'] = df['Prediction'] > threshold
    df['Entry_Price'] = df['Close']
    df['Exit_Price'] = df['Close'].shift(-shift)
    df['PnL'] = np.where(df['Buy_Signal'], df['Exit_Price'] - df['Entry_Price'], 0)
    df['Exit_Time'] = df['Date'].shift(-shift)
    trades = df[df['Buy_Signal']].copy()
    trades = trades.dropna(subset=['Exit_Price'])
    trade_log = trades[['Date', 'Exit_Time','Entry_Price', 'Exit_Price', 'PnL', 'Prediction']].copy()
    trade_log['Date'] = trade_log['Date'].astype(str)
    trade_log['Exit_Time'] = trade_log['Exit_Time'].astype(str)

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
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm
from xgboost import XGBClassifier
import pandas as pd

def backtest_strategy_rolling(
    df,
    shift=5,
    model_type='logit',
    features=None,
    cutoff=0.999999,
    min_train_size=200,
    transaction_cost=0.002,
    retrain_every=5,
    verbose=True
):
    """
    Faster, realistic rolling backtest that:
    - Trains every `retrain_every` steps
    - Predicts one day ahead
    - Simulates forward trade (t to t+shift)
    - Applies transaction cost

    Returns:
        DataFrame with trade log
    """
    assert features is not None, "Must provide a list of features."

    trades = []
    df = df.copy().reset_index(drop=True)
    df = df.replace([float('inf'), float('-inf')], pd.NA).dropna()
    df = add_target(df, shift=shift).dropna().reset_index(drop=True)

    model = None

    for t in tqdm(range(min_train_size, len(df) - shift), desc="Rolling Backtest"):
        # Slice train/test
        train_data = df.iloc[:t]
        # Features/labels
        X_train = train_data[features]
        y_train = train_data['Target']
        X_test = df.loc[[t], features]  # needs to be 2D for prediction
        # Retrain model every few steps
        retrain = ((t - min_train_size) % retrain_every == 0) or (model is None)
        try:
            if model_type == 'logit':
                if retrain:
                    try:
                        X_train_const = sm.add_constant(X_train, has_constant='add')
                        model = sm.Logit(y_train, X_train_const).fit(disp=0)
                        model_columns = X_train_const.columns
                    except Exception as e:
                        if verbose:
                            print(f"Logit training failed at t={t}")
                            import traceback
                            traceback.print_exc()
                        continue
                try:
                    X_test_const = sm.add_constant(X_test, has_constant='add')
                    X_test_const = X_test_const.reindex(columns=model_columns, fill_value=0)
                    prob = model.predict(X_test_const).iloc[0]
                except Exception as e:
                    if verbose:
                        print(f"Logit prediction failed at t={t}")
                        import traceback
                        traceback.print_exc()
                    continue

            elif model_type == 'xgb':
                if retrain:
                    model = XGBClassifier(
                        eval_metric='logloss',
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                prob = model.predict_proba(X_test)[0, 1]

            else:
                raise ValueError("Invalid model_type")

        except Exception as e:
            if verbose:
                print(f"Model error at t={t}: {e}")
            continue

        # Optional: print a few early probabilities
        if verbose and t < min_train_size + 10:
            print(f"t={t}, prob={prob:.4f}")

        if prob > cutoff:
            entry_price = df.at[t, 'Close']
            exit_price = df.at[t + shift, 'Close']
            pnl = exit_price - entry_price - (entry_price * transaction_cost)

            trades.append({
                'Date': df.at[t, 'Date'],
                'Entry_Price': entry_price,
                'Exit_Price': exit_price,
                'PnL': pnl,
                'Prob': prob
            })

    plt.close('all')  # clear figures to avoid memory warnings

    return pd.DataFrame(trades)


def explore_shift_auc(train, val):
    print(f"Exploring AUC over SHIFT range for {TICKER} on {INTERVAL} interval\n")
    results = []
    for shift in SHIFT_RANGE:
        try:
            train_target = add_target(train.copy(), shift)
            val_target = add_target(val.copy(), shift)

            X_train = train_target[STRATEGY]
            y_train = train_target['Target']
            X_val = val_target[STRATEGY]
            y_val = val_target['Target']

            # Skip if only one class in validation target
            if len(np.unique(y_val)) < 2:
                print(f"Shift {shift}: Skipped due to only one class in validation set")
                continue

            model = XGBClassifier(
                eval_metric='logloss',
                random_state=42,
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1
            )
            model.fit(X_train, y_train)
            y_val_prob = model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, y_val_prob)
            print(f"Shift: {shift:2d} | AUC: {auc_score:.4f}")
            results.append({'Shift': shift, 'AUC': auc_score})
        except Exception as e:
            print(f"Shift {shift}: Skipped due to error ({e})")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='AUC', ascending=False).reset_index(drop=True)
    return results_df

# Study dataset using provided train/val split to select shift

def study_dataset(train, val, full_df):
    global OPTIMAL_SHIFT

    # Explore AUC over multiple shifts
    results_df = explore_shift_auc(train, val)
    plot_df = results_df.sort_values(by='Shift')

    # Plot AUC vs. Shift
    plt.figure(figsize=(10, 5))
    plt.plot(plot_df['Shift'], plot_df['AUC'], marker='o')
    plt.title(f'AUC by SHIFT for {TICKER} ({INTERVAL})')
    plt.xlabel('SHIFT')
    plt.ylabel('AUC Score')
    plt.grid(True)

    # Set optimal shift globally for future studies
    OPTIMAL_SHIFT = int(results_df.iloc[0]['Shift'])
    print(f"\nOptimal SHIFT based on validation AUC: {OPTIMAL_SHIFT}")

    # Final model trained on full data

    df_final = add_target(full_df.copy(), shift=OPTIMAL_SHIFT)
    # df_final, y_final, y_pred_prob = generate_regression_output(df_final)

    # plot_prediction_distribution(y_pred_prob)
    # add_roc_plot(y_final, y_pred_prob, title=f'ROC Curve (SHIFT = {OPTIMAL_SHIFT})')
    # add_confusion_matrix(df_final)
    return df_final, results_df


def evaluate_on_test(train_df: pd.DataFrame, val_df: pd.DataFrame,
                     test_df: pd.DataFrame, shift: int,
                     features: list, cutoff=CUTOFF, model="xgb"):
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
        X_comb_const = sm.add_constant(X_comb)
        X_test_const = sm.add_constant(X_test)
        final_model = sm.Logit(y_comb, X_comb_const).fit(disp=0)
        y_pred_prob = final_model.predict(X_test_const)

    else:
        raise ValueError("model must be either 'xgb' or 'logit'")

    test_t['Prediction'] = (y_pred_prob > cutoff).astype(int)

    return test_t, y_test, y_pred_prob



def add_indicators(df):
    df = add_MACD(df)
    df= add_MFI(df)
    df = add_BB(df)
    df=add_RSI(df)

    return df

# Main execution
if __name__ == '__main__':
    df = get_data()
    # Split once
    train, val, test = train_val_test_split(df)
    train=add_indicators(train)
    val=add_indicators(val)
    test=add_indicators(test)

    # Determine optimal shift using train/val only
    df = add_indicators(df)
    df_final, results_df = study_dataset(train, val, df)
    # Train model with optimal shift on train, evaluate on test
    test_df, y_test, y_prob = evaluate_on_test(train, val, test, shift=OPTIMAL_SHIFT, features=STRATEGY,cutoff=CUTOFF,
                                               model="logit")
    plot_prediction_distribution(y_prob)
    add_roc_plot(y_test, y_prob, title=f'ROC Curve (Test Set, SHIFT={OPTIMAL_SHIFT})')
    add_confusion_matrix(test_df)

    # Backtest strategy
    # backtest_strategy(test_df, shift=OPTIMAL_SHIFT, threshold=CUTOFF, original_df=test)

    rolling_trades = backtest_strategy_rolling(df,shift=OPTIMAL_SHIFT,model_type='logit', features=STRATEGY,
                                               cutoff=0.5,transaction_cost=0.002
)
    
print(rolling_trades.head())
print("Total Trades:", len(rolling_trades))
print("Total Return:", rolling_trades['PnL'].sum())
print("Win Rate:", (rolling_trades['PnL'] > 0).mean())
