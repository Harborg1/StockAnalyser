import numpy as np
import pandas as pd
import yfinance as yf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import json
from datetime import datetime

password = os.environ.get("EMAIL_PASSWORD")
sender_email = os.environ.get("EMAIL_SENDER")
receiver_email = os.environ.get("EMAIL_RECIEVER")

# Keep track of signals sent today
SIGNAL_LOG_FILE = os.path.join("json_folder","signals_sent.json")

if os.path.exists(SIGNAL_LOG_FILE):
    try:
        with open(SIGNAL_LOG_FILE, "r") as f:
            signals_sent_today = json.load(f)
    except json.JSONDecodeError:
        # File exists but is empty or corrupted → reset it
        print(f"⚠️ Warning: Could not load {SIGNAL_LOG_FILE} — file was empty or corrupted. Resetting.")
        signals_sent_today = {}
else:
    signals_sent_today = {}

# Get today's date as string
today_str = datetime.now().strftime("%Y-%m-%d")

# If this is a new day, reset the signals log
if signals_sent_today.get("date") != today_str:
    signals_sent_today = {"date": today_str, "signals": {}}

with open(SIGNAL_LOG_FILE, "w") as f:
        json.dump(signals_sent_today, f)

def send_trading_signal(ticker):
    stock = yf.download(ticker)
    stock['day_return_pct'] = stock['Close'].pct_change() * 100
    stock['cum_return_3d'] = stock['day_return_pct'].rolling(window=3).sum()
    stock.dropna(inplace=True)

    latest_day_return = stock['day_return_pct'].iloc[-1]
    latest_cum_return_3d = stock['cum_return_3d'].iloc[-1]
    latest_close = round(stock['Close'].iloc[-1], 2)

    # Priority logic
    signal_text = None

    if latest_day_return >= 10:
        signal_text = f"UP10_TODAY"
    elif latest_cum_return_3d >= 10:
        signal_text = f"UP10_3DAY"
    elif latest_day_return <= -10:
        signal_text = f"DOWN10_TODAY"
    elif latest_cum_return_3d <= -10:
        signal_text = f"DOWN10_3DAY"

    # Check if signal was already sent
    if signal_text:
        ticker_signals = signals_sent_today["signals"].get(ticker, [])
        print("Ticker signals is:",ticker_signals)
        print("Signal text is",signal_text)
        if signal_text in ticker_signals:
            print(f"⏭️ Signal '{signal_text}' for {ticker} already sent today. Skipping.")
            return
        
        # Else: send email
        subject = f"📈 Trading Signal Alert for {ticker}"
        body = f"""Hi,

A trading signal has been detected for {ticker}.

{signal_text}

🔹 Latest closing price: {latest_close} USD

"""
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, msg.as_string())
            print(f"✅ Email sent for {ticker}: {signal_text}")

            # Mark signal as sent
            ticker_signals.append(signal_text)
            signals_sent_today["signals"][ticker] = ticker_signals

            # Save updated signals log
            with open(SIGNAL_LOG_FILE, "w") as f:
                json.dump(signals_sent_today, f)

        except Exception as e:
            print(f"❌ Failed to send email for {ticker}: {e}")
    else:
        print(f"No signal detected for {ticker}. No email sent.\n")
        print(latest_cum_return_3d)

# Current portfolio
portfolio = ["NOVO-B.CO", "TSLA", "CLSK", "NVDA"]

for stock in portfolio:
    send_trading_signal(stock)
