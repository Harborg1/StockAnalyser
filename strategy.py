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
import pandas as pd

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
    

    def get_indicators(self,stock):
            """
            Get SMA20, SMA50, SMA200, ATR(14), ATR%, RSI(14), MACD, Avg Vol, RVOL, Gap %.
            """
            df = yf.download(stock, period="1y", interval="1d", progress=False, auto_adjust=False).dropna()
            if df.empty:
                return {}

            close = df["Close"][stock] if isinstance(df.columns, pd.MultiIndex) else df["Close"]

            # SMA
            sma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None
            sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
            sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

            # ATR(14)
            high = df["High"][stock] if isinstance(df.columns, pd.MultiIndex) else df["High"]
            low = df["Low"][stock] if isinstance(df.columns, pd.MultiIndex) else df["Low"]
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean().iloc[-1]
            atr_pct = (atr14 / close.iloc[-1] * 100) if atr14 else None

            # RSI(14)
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0.0).rolling(14).mean()
            rs = gain / loss
            rsi14 = 100 - (100 / (1 + rs.iloc[-1])) if loss.iloc[-1] != 0 else 100

            # MACD (12,26,9)
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - signal_line

            # Volume metrics
            vol = df["Volume"][stock] if isinstance(df.columns, pd.MultiIndex) else df["Volume"]
            avg_vol20 = vol.tail(20).mean()
            last_vol = vol.iloc[-1]
            rvol = last_vol / avg_vol20 if pd.notna(avg_vol20) and avg_vol20 != 0 else None

            def to_float_2(v): return float(round(v, 2))

            return {
                "sma20": to_float_2(sma20),
                "sma50": to_float_2(sma50),
                "sma200": to_float_2(sma200),
                "atr14": to_float_2(atr14),
                "atr_pct": to_float_2(atr_pct),
                "rsi14": to_float_2(rsi14),
                "macd_line": to_float_2(macd_line.iloc[-1]),
                "signal_line": to_float_2(signal_line.iloc[-1]),
                "macd_hist": to_float_2(macd_hist.iloc[-1]),
                "avg_vol20": to_float_2(avg_vol20),
                "rvol": to_float_2(rvol),
            }


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

def ask_openai_for_strategy(client, stock, closes, indicators, pre_market):
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
- Last 5 days Open, High, Low, Close: {closes}
- Pre-market price: {pre_market}
- 20-day moving average: {indicators["sma20"]}
- 50-day moving average: {indicators["sma50"]}
- 200-day moving average: {indicators["sma200"]}
- Average True Range the past 14 days: {indicators["atr14"]}
- Average True Range Percentage: {indicators["atr_pct"]}
- RSI the past 14 days: {indicators["rsi14"]}
- MACD line: {indicators["macd_line"]}
- Signal line: {indicators["signal_line"]}
- MACD history: {indicators["macd_hist"]}
- Average volume the past 20 days: {indicators["avg_vol20"]}
- Relative volume the latest close: {indicators["rvol"]}

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
    indicators = s.get_indicators(stock)
    strategy = ask_openai_for_strategy(client, stock, closes, indicators, pre_market)

    body = f"""📈 AI-Generated Trading Strategy for {stock}\n\n{strategy}"""
    send_email(f"📊 Daily Strategy for {stock}", body, sender_email, receiver_email, password)
    print("✅ Strategy email sent.")
