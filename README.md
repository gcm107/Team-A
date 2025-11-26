# Team-A
Team A Group Project


The goal of this project is to discover short-term stock market “micro-regimes” from recent price/volume features using unsupervised learning and then test, with supervised models, whether those regimes and their transitions help predict future returns or volatility.

## Exploration
Start with the `data-exploration.ipynb`, which shows how to load and explore the dataset (stored as a `.parquet` file). You can download the dataset from [Team A Dataset](https://sooners-my.sharepoint.com/:f:/g/personal/grason_c_mcdowell-1_ou_edu/EhDr2-QJqhdGryc_EcrvAGsBPVKuJQu_wBhOCPNeS3v2Hg?e=Slxi8G).

The dataset should be placed in `data/historical/all_stocks_historical.parquet`. If you download from onedrive please place it there. 


## Stock Data
We pulled historical stock price data using yfinance. Started with 7000+ tickers from `data/all_symbols_2025_11_13-1027.csv`.

### What's in here

Daily OHLCV data (Open, High, Low, Close, Volume) for NASDAQ and NYSE stocks. Close prices are already adjusted for splits and dividends

**Quick stats:**
- 7,144 tickers processed
- 7,094 successfully downloaded (50 failed - probably delisted or no data)
- Date range: 2015-01-01 to present
- 141,880 rows total
- Saved as Parquet (ZSTD compressed)

### How we made it

1. **Got the tickers**
   -  CSV with tickers (`data/all_symbols_2025_11_13-1027.csv`)
   - removed exchange prefixes (like "NASDAQ:AAPL" → "AAPL")
   - checked and removed duplicates


2. **Downloaded the data**
   - Used yfinance to grab historical prices


3. **Saved it**
   - parquet file: `data/historical/all_stocks_historical.parquet`
   - metadata `data/historical/download_metadata.json`

### Columns

- `ticker` - Stock symbol (like "AAPL")
- `date` - Trading date
- `open` - Opening price
- `high` - Highest price that day
- `low` - Lowest price that day
- `close` - Closing price (already adjusted for splits/dividends)
- `volume` - How many shares traded

### notes
- files used to download data are in the `qualifier/utils/` folder. We shouldn't use those anymore unless we need a new dataset. lets keep them in the repo though.
- We removed duplicate tickers (same symbol on different exchanges)
- Some downloads failed (delisted stocks, etc.)
- Close prices are adjusted