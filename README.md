# Team-A
Team A Group Project


The goal of this project is to use simple statistics based on daily returns to help us (1) group days into different market “modes” and (2) build machine learning models that predict next week's movement and returns

## Exploration
Start with the `data-exploration.ipynb`, which shows how to load and explore the dataset (stored as a `.parquet` file). You can download the dataset from 

## Stock Data
We pulled historical stock price data using yfinance for tickers and baseline:
```
tickers = ["TSLA", "JPM", "MSFT", "AAPL", "XOM"]
benchmark = "SPY"
```

### What's in here

Daily OHLCV data (Open, High, Low, Close, Volume) for NASDAQ and NYSE stocks. Close prices are already adjusted for splits and dividends

### Columns

- `ticker` - Stock symbol (like "AAPL")
- `date` - Trading date
- `open` - Opening price
- `high` - Highest price that day
- `low` - Lowest price that day
- `close` - Closing price (already adjusted for splits/dividends)
- `volume` - How many shares traded
