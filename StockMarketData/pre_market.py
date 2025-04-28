import yfinance as yf
import datetime
import contextlib
import os

def get_pre_market_price_ticker(ticker: str) -> float:
    date = datetime.datetime.now().date()
    try:
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                data = yf.download(
                    ticker,
                    start=f"{date}",
                    end=f"{date + datetime.timedelta(days=1)}",
                    interval="1m",
                    prepost=True,
                    progress=False
                )
        if data.empty:
            print("data was empty..")
            return None
        return data["Close"].iloc[-1][ticker]
    except Exception:
        print("Could not retrieve data.")
        return None


# last_close = get_pre_market_price_ticker("CLSK")
# print(last_close)
