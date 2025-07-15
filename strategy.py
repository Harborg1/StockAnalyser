import os
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from openai import OpenAI
from stocks.base_reader import MarketReaderBase
from auxillary.pre_market import get_pre_market_price_ticker
import yfinance as yf

load_dotenv("passcodes.env")

# Load credentials from env
password = os.environ.get("EMAIL_PASSWORD")
sender_email = os.environ.get("EMAIL_SENDER")
receiver_email = os.environ.get("EMAIL_RECIEVER")
api_key = os.environ.get("OPENAI_API_KEY") or os.getenv("CHAT_GPT_KEY")

# Validate
if not all([password, sender_email, receiver_email, api_key]):
    raise EnvironmentError("Missing one or more required environment variables.")

client = OpenAI(api_key=api_key)

class Strategy(MarketReaderBase):
    def __init__(self, stock):
        super().__init__()
        self.stock = stock
        self.ma20 = self.get_moving_average(self.start_date, self.end_date, stock, True)
        self.ma50 = self.get_moving_average(self.start_date, self.end_date, stock, False)

    def download_data_stock(self, start_date, end_date, stock):
        ticker = yf.download(stock, start=start_date, end=end_date)
        last_5 = ticker.dropna().tail(5)
        return [
            {
                "date": str(date.date()),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2)
            }
            for date, row in last_5.iterrows()
        ]

def ask_openai_for_strategy(client, stock, closes, ma20, ma50, pre_market):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional stock trader and financial analyst with deep expertise in short-term trading strategies, "
                "technical analysis, and intraday decision-making. Your job is to provide clear, actionable advice for the given trading day."
            )
        },
        {
            "role": "user",
            "content": f"""
Analyze the stock data for {stock} and recommend a trading strategy (buy, hold, or sell) for today. Assume I am trading at the market open and want to optimize for risk-reward.

**Data:**
- Last 5 days OHLC: {closes}
- 20-day moving average: {ma20}
- 50-day moving average: {ma50}
- Pre-market price: {pre_market}

**Requirements:**
1. Recommend an action: **buy**, **hold**, or **sell**.
2. Suggest a **stop-loss** level to minimize risk.
3. Suggest a **take-profit** level or range for today.
4. Suggest how much of the total position to take profit at that level (e.g., 30%, 50%), and whether to **leave a portion to run** with a trailing stop.
5. Justify each suggestion briefly using the data and principles of technical analysis.
"""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return response.choices[0].message.content

def send_email(subject, body, sender, receiver, password):
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())

if __name__ == "__main__":
    stock = "CLSK"
    s = Strategy(stock)
    closes = s.download_data_stock(s.start_date, s.end_date, stock)
    pre_market = get_pre_market_price_ticker(stock)
    strategy = ask_openai_for_strategy(client, stock, closes, s.ma20, s.ma50, pre_market)

    body = f"""📈 AI-Generated Trading Strategy for {stock}\n\n{strategy}"""
    send_email(f"📊 Daily Strategy for {stock}", body, sender_email, receiver_email, password)
    print("✅ Strategy email sent.")
