import os
import yfinance as yf
from openai import OpenAI
from dotenv import load_dotenv
from stocks.base_reader import MarketReaderBase
from auxillary.pre_market import get_pre_market_price_ticker
load_dotenv("passcodes.env")

# Load API key
# Try to load from OPENAI_API_KEY first, fallback to local .env CHAT_GPT_KEY
api_key = os.environ.get("OPENAI_API_KEY") or os.getenv("CHAT_GPT_KEY")

# Ensure it's not missing
if not api_key:
    raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY or CHAT_GPT_KEY in your environment or .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

class Strategy(MarketReaderBase):
    def __init__(self, stock):
        super().__init__()
        self.stock = stock
        self.ma20 = self.get_moving_average(self.start_date, self.end_date, stock, True)
        self.ma50 = self.get_moving_average(self.start_date, self.end_date, stock, False)

    def download_data_stock(self, start_date, end_date, stock):
        ticker = yf.download(stock, start=start_date, end=end_date)

        # Get the last 5 rows, drop NaNs
        last_5 = ticker.dropna().tail(5)
        print(last_5)

        # Return a list of dicts with OHLC values rounded to 2 decimal places
        return [
            {
                "date": str(date.date()),
                "open": round(float(row["Open"][stock]), 2),
                "high": round(float(row["High"][stock]), 2),
                "low": round(float(row["Low"][stock]), 2),
                "close": round(float(row["Close"][stock]), 2)
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

Act as an experienced trader who understands volatility, breakouts, and the importance of position management.
"""
}

    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    
    return response.choices[0].message.content

# Main script
if __name__ == "__main__":
    stock = "CLSK"
    s = Strategy(stock)
    closes = s.download_data_stock(s.start_date, s.end_date, stock)
    pre_market = get_pre_market_price_ticker(stock)

    print("The pre-market price is:", pre_market)
    print("The 5 latest closing prices were:", closes)
    print("The 20-day moving average is:", s.ma20)
    print("The 50-day moving average is:", s.ma50)

    strategy = ask_openai_for_strategy(client, stock, closes, s.ma20, s.ma50, pre_market)
    print("\n📈 AI-Generated Strategy:\n", strategy)
