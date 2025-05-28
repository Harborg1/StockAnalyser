import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
load_dotenv("passcodes.env")

password= os.getenv("EMAIL_PASSWORD")

# Fetch data
gld = yf.download('TSLA')
gld.columns = gld.columns.get_level_values(0)

# Add day index
gld['day'] = np.arange(1, len(gld) + 1)
gld = gld[['day', 'Open', 'High', 'Low', 'Close']]

# Moving averages with shift to avoid lookahead bias
gld['9-day'] = gld['Close'].rolling(9).mean().shift()
gld['21-day'] = gld['Close'].rolling(21).mean().shift()

# Generate trading signals
gld['signal'] = np.where(gld['9-day'] > gld['21-day'], 1, 0)
gld['signal'] = np.where(gld['9-day'] < gld['21-day'], -1, gld['signal'])

# Drop NaNs
gld.dropna(inplace=True)

# Calculate log returns and strategy performance
gld['return'] = np.log(gld['Close']).diff()
gld['system_return'] = gld['signal'] * gld['return']
gld['entry'] = gld['signal'].diff()

# Filter data for 2025
gld_2025 = gld[gld.index >= '2025-01-01']

# Plot 2025 signals and prices
plt.figure(figsize=(12, 6))
plt.grid(True, alpha=.3)
plt.plot(gld_2025['Close'], label='CLSK')
plt.plot(gld_2025['9-day'], label='9-day')
plt.plot(gld_2025['21-day'], label='21-day')
plt.plot(gld_2025.loc[gld_2025.entry == 2].index, gld_2025['9-day'][gld_2025.entry == 2], '^',
         color='g', markersize=12, label='Buy')
plt.plot(gld_2025.loc[gld_2025.entry == -2].index, gld_2025['21-day'][gld_2025.entry == -2], 'v',
         color='r', markersize=12, label='Sell')
plt.legend(loc=2)
plt.title("CLSK Price and Trading Signals - 2025")
plt.show()

# Plot cumulative returns for 2025
plt.figure(figsize=(12, 6))
plt.plot(np.exp(gld_2025['return'].cumsum()), label='Buy/Hold')
plt.plot(np.exp(gld_2025['system_return'].cumsum()), label='System')
plt.legend(loc=2)
plt.grid(True, alpha=.3)
plt.title("Cumulative Returns in 2025")
plt.show()

# Print final returns for 2025
buy_hold_return = np.exp(gld_2025['return'].sum()) - 1
system_return = np.exp(gld_2025['system_return'].sum()) - 1
print("2025 Buy & Hold return:", buy_hold_return)
print("2025 System return:", system_return)

# Check signal at the latest date
if gld['9-day'].iloc[-1] > gld['21-day'].iloc[-1]:
    sender_email = "christian1234t4556565@gmail.com"  # 🔒 Replace with your email
    receiver_email = "caharborg@gmail.com"
    code = password  # 🔒 Replace with app-specific password (not your actual Gmail password)

    subject = "Trading Signal Alert: 9-day MA > 21-day MA"
    body = f"""\
    Hello,
    A bullish signal has been detected:
    - 9-day MA: {gld['9-day'].iloc[-1]:.2f}
    - 21-day MA: {gld['21-day'].iloc[-1]:.2f}

    Consider reviewing TSLA.

    Regards,
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
