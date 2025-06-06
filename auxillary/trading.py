import numpy as np
import pandas as pd
import yfinance as yf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

password = os.environ.get("EMAIL_PASSWORD")
sender_email = os.environ.get("EMAIL_SENDER")
receiver_email =  os.environ.get("EMAIL_RECIEVER")

def send_trading_signal(ticker):
    stock = yf.download(ticker)
    stock['day_return_pct'] = stock['Close'].pct_change() * 100
    stock['cum_return_3d'] = stock['day_return_pct'].rolling(window=3).sum()
    stock.dropna(inplace=True)

    latest_day_return = stock['day_return_pct'].iloc[-1]
    latest_cum_return_3d = stock['cum_return_3d'].iloc[-1]
    latest_close = round(stock['Close'].iloc[-1], 2)

    # Priority logic
    signal = None

    if latest_day_return >= 10:
        signal = f"{ticker}: Up {latest_day_return:.2f}% today — consider SELLING 30%"
    elif latest_cum_return_3d >= 10:
        signal = f"{ticker}: Up {latest_cum_return_3d:.2f}% in last 3 days — consider SELLING 30%"
    elif latest_day_return <= -10:
        signal = f"{ticker}: Down {latest_day_return:.2f}% today — consider BUYING 30%"
    elif latest_cum_return_3d <= -10:
        signal = f"{ticker}: Down {latest_cum_return_3d:.2f}% in last 3 days — consider BUYING 30%"

    # Send email if signal exists
    if signal:
        subject = f"📈 Trading Signal Alert for {ticker}"
        body = f"""Hi,

A trading signal has been detected for {ticker}.

{signal}

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
            print(f"✅ Email sent for {ticker}: {signal}")
        except Exception as e:
            print(f"❌ Failed to send email for {ticker}: {e}")
    else:
        print(f"No signal detected for {ticker}. No email sent.")

# Example portfolio
portfolio = ["NOVO-B.CO", "TSLA", "CLSK", "NVDA"]

for stock in portfolio:
    send_trading_signal(stock)
