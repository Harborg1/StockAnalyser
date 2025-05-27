import yfinance as yf
import datetime
import contextlib
import os
"""This function tries to get the pre market price and returns it as a float value if it exists.
   If no data was available, the method returns None."""
def get_pre_market_price_ticker(ticker: str) -> float | None:
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
            return None
        return data["Close"].iloc[-1][ticker]
    except Exception:
        print("Could not retrieve data.")
        return None
    

    
    
    


