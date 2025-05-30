import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# In GitHub Actions, this comes from the environment
password = os.environ.get("EMAIL_PASSWORD")

sender_email = "christian1234t4556565@gmail.com"
receiver_email = "caharborg@gmail.com"

# Fetch data
stock = yf.download('TSLA')
stock.columns = stock.columns.get_level_values(0)

# Add day index
stock['day'] = np.arange(1, len(stock) + 1)
stock = stock[['day', 'Open', 'High', 'Low', 'Close']]

# Moving averages with shift to avoid lookahead bias
stock['9-day'] = stock['Close'].rolling(9).mean().shift()
stock['21-day'] = stock['Close'].rolling(21).mean().shift()

# Generate trading signals
stock['signal'] = np.where(stock['9-day'] > stock['21-day'], 1, 0)
stock['signal'] = np.where(stock['9-day'] < stock['21-day'], -1, stock['signal'])

# Drop NaNs
stock.dropna(inplace=True)

# Calculate log returns and strategy performance
stock['return'] = np.log(stock['Close']).diff()
stock['system_return'] = stock['signal'] * stock['return']
stock['entry'] = stock['signal'].diff()

# Filter data for 2025
stock_2025 = stock[stock.index >= '2025-01-01']

# Plot 2025 signals and prices
plt.figure(figsize=(12, 6))
plt.grid(True, alpha=.3)
plt.plot(stock_2025['Close'], label='CLSK')
plt.plot(stock_2025['9-day'], label='9-day')
plt.plot(stock_2025['21-day'], label='21-day')
plt.plot(stock_2025.loc[stock_2025.entry == 2].index, stock_2025['9-day'][stock_2025.entry == 2], '^',
         color='g', markersize=12, label='Buy')
plt.plot(stock_2025.loc[stock_2025.entry == -2].index, stock_2025['21-day'][stock_2025.entry == -2], 'v',
         color='r', markersize=12, label='Sell')
plt.legend(loc=2)
plt.title("CLSK Price and Trading Signals - 2025")
plt.show()

# Plot cumulative returns for 2025
plt.figure(figsize=(12, 6))
plt.plot(np.exp(stock_2025['return'].cumsum()), label='Buy/Hold')
plt.plot(np.exp(stock_2025['system_return'].cumsum()), label='System')
plt.legend(loc=2)
plt.grid(True, alpha=.3)
plt.title("Cumulative Returns in 2025")
plt.show()

# Print final returns for 2025
buy_hold_return = np.exp(stock_2025['return'].sum()) - 1
system_return = np.exp(stock_2025['system_return'].sum()) - 1
print("2025 Buy & Hold return:", buy_hold_return)
print("2025 System return:", system_return)

# Check signal at the latest date
if stock['9-day'].iloc[-1] > stock['21-day'].iloc[-1]:
    code = password 

    subject = "📈 Trading Signal Alert for TSLA: Bullish Crossover Detected"

    latest_close = round(stock['Close'].iloc[-1], 2)
    ma_9 = round(stock['9-day'].iloc[-1], 2)
    ma_21 = round(stock['21-day'].iloc[-1], 2)

    body = f"""Hi,

    A *bullish crossover* has just been detected for TSLA.

    🔔 The 9-day moving average ({ma_9}) is now higher than the 21-day moving average ({ma_21}).

    🔹 Latest closing price: {latest_close} USD  
    🔹 Signal: Potential buy indication

    This could be an opportunity worth watching, depending on your trading strategy.

    Best regards,  
    Your Python Script
    """

    # Create the email
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, code)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("✅ Email sent successfully.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
else:
    print("ℹ️ No bullish crossover detected. No email sent.")