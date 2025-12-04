"""
functions to fetch stock data using yfinance and prepare features
the same we do in data-exploration.ipynb
"""

import pandas as pd
import yfinance as yf
from typing import Optional, Dict


def prepare_features_from_ticker(
    ticker: str,
    date: Optional[str] = None,
    start_date: str = "2015-01-01"
) -> Dict[str, float]:
    
    # Fetch stock data and prepare features for prediction.
    
    benchmark = "SPY"
    
    # download ticker data
    ticker_data = yf.download(
        ticker,
        start=start_date,
        end=None,
        auto_adjust=True,
        progress=False
    )
    
    if ticker_data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    
    # download SPY data
    spy_data = yf.download(
        benchmark,
        start=start_date,
        end=None,
        auto_adjust=True,
        progress=False
    )
    
    if spy_data.empty:
        raise ValueError(f"No data found for benchmark: {benchmark}")
    
    # get closing prices
    prices_stock = ticker_data["Close"].copy()
    prices_spy = spy_data["Close"].copy()
    
    # combine and drop missing values
    prices = pd.concat([prices_stock, prices_spy], axis=1)
    prices.columns = ["price", "spy_price"]
    prices = prices.dropna(how="any")
    
    if len(prices) < 5:
        raise ValueError(f"Insufficient data for ticker {ticker}. Need at least 5 days.")
    
    # compute daily returns
    prices["daily_return"] = prices["price"].pct_change()
    prices["spy_return"] = prices["spy_price"].pct_change()
    
    # compute rolling 5-day statistics
    prices["mean_5d"] = prices["daily_return"].rolling(window=5, min_periods=5).mean()
    prices["vol_5d"] = prices["daily_return"].rolling(window=5, min_periods=5).std()
    
    # drop rows containing NaN (first 5 days don't have rolling stats and creates nans)
    prices = prices.dropna(subset=["daily_return", "mean_5d", "vol_5d", "spy_return"])
    
    if prices.empty:
        raise ValueError(f"No data for ticker {ticker}. Need at least 5 days after computing rolling stats.")
    
    # select date
    if date:
        try:
            target_date = pd.to_datetime(date)
            if target_date not in prices.index:
                #nearest date
                nearest_idx = prices.index.get_indexer([target_date], method="nearest")[0]
                if nearest_idx == -1:
                    raise ValueError(f"date {date} not found in data range")
                actual_date = prices.index[nearest_idx]
            else:
                actual_date = target_date
        except Exception as e:
            raise ValueError(f" date format is wrong: {date}. Use YYYY-MM-DD format.")
    else:
        # most recent date
        actual_date = prices.index[-1]
    
    # extract features for the selected date
    row = prices.loc[actual_date]
    
    return {
        "daily_return": float(row["daily_return"]),
        "mean_5d": float(row["mean_5d"]),
        "vol_5d": float(row["vol_5d"]),
        "spy_return": float(row["spy_return"]),
        "date": actual_date.strftime("%Y-%m-%d")
    }

