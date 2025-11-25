"""
downloads stock data and saves it to parquet.
reads symbols csv, removes duplicates, and grabs ohlcv data via yfinance.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field, field_validator


def get_project_root() -> Path:
    """gets project root, 3 folders up from here."""
    return Path(__file__).parent.parent.parent

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SymbolConfig(BaseModel):
    """config for downloading stock data."""

    input_csv: Path = Field(default_factory=lambda: get_project_root() / "data" / "all_symbols_2025_11_13-1027.csv")
    output_dir: Path = Field(default_factory=lambda: get_project_root() / "data" / "historical")
    start_date: str | None = Field(default=None, description="start date yyyy-mm-dd, ignored if period is set")
    end_date: str | None = Field(default=None, description="end date yyyy-mm-dd, none means today, ignored if period is set")
    period: str | None = Field(
        default=None, description="how far back: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'"
    )
    batch_size: int = Field(default=50, ge=1, le=100)
    max_retries: int = Field(default=3, ge=1)
    max_tickers: int | None = Field(
        default=None, description="max number of tickers to grab, none = all"
    )

    @field_validator("input_csv", "output_dir")
    @classmethod
    def validate_paths(cls, v: Path) -> Path:
        """make sure paths are path objects."""
        return Path(v)


def extract_unique_tickers(csv_path: Path) -> pd.DataFrame:
    """
    pulls unique tickers from the csv, strips exchange prefix.
    returns dataframe with unique tickers and exchanges.
    """
    logger.info(f"Reading symbols from {csv_path}")
    df = pd.read_csv(csv_path)

    # grab ticker part from "EXCHANGE:TICKER"
    df["ticker"] = df["symbol"].str.split(":").str[-1]

    # keep first one if there's duplicates
    df_unique = df.drop_duplicates(subset=["ticker"], keep="first").copy()

    logger.info(
        f"Found {len(df)} total symbols, {df['ticker'].nunique()} unique tickers "
        f"({len(df) - df['ticker'].nunique()} duplicates removed)"
    )

    return df_unique[["ticker", "exchange"]].reset_index(drop=True)


def download_batch(
    tickers: list[str],
    config: SymbolConfig,
    start_idx: int = 0,
) -> pd.DataFrame | None:
    """
    downloads data for a bunch of tickers at once using yfinance.
    returns dataframe in long format or none if it fails.
    """
    if not tickers:
        return None

    total = len(tickers)
    logger.info(f"Downloading batch of {total} tickers (indices {start_idx + 1}-{start_idx + total})...")

    try:
        # download all tickers in parallel
        tickers_obj = yf.Tickers(" ".join(tickers))
        
        # get history - comes back as multiindex (metric, ticker)
        # defaults to "max" to get everything
        if config.period:
            hist = tickers_obj.history(period=config.period)
        elif config.start_date and config.end_date is None:
            # just start date, goes to today
            hist = tickers_obj.history(start=config.start_date)
        elif config.start_date:
            # both dates given
            hist = tickers_obj.history(start=config.start_date, end=config.end_date)
        else:
            # default to max
            hist = tickers_obj.history(period="max")

        if hist.empty:
            logger.warning(f"No data downloaded for batch of {total} tickers")
            return None

        # reshape to long format
        results = []
        
        # check if we got multiindex columns
        if isinstance(hist.columns, pd.MultiIndex):
            # multiple tickers - extract each one
            available_tickers = hist.columns.levels[1] if len(hist.columns.levels) > 1 else []
            
            for ticker in tickers:
                if ticker not in available_tickers:
                    logger.warning(f"No data available for {ticker}")
                    continue

                # grab columns for this ticker
                ticker_cols = [col for col in hist.columns if col[1] == ticker]
                ticker_data = hist[ticker_cols].copy()
                
                # flatten column names
                ticker_data.columns = [col[0] for col in ticker_data.columns]
                ticker_data = ticker_data.reset_index()
                ticker_data["ticker"] = ticker
                results.append(ticker_data)
        else:
            # single ticker - already flat
            ticker = tickers[0]
            ticker_data = hist.copy()
            ticker_data = ticker_data.reset_index()
            ticker_data["ticker"] = ticker
            results.append(ticker_data)

        if not results:
            logger.warning(f"No valid data in batch")
            return None

        df = pd.concat(results, ignore_index=True)

        # rename to standard format
        df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        # make date datetime and strip timezone
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        # keep only ohlcv + ticker
        cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]]

        logger.info(f"Successfully downloaded {df['ticker'].nunique()} tickers with {len(df):,} total rows")
        return df

    except Exception as e:
        logger.error(f"Failed to download batch: {e}")
        return None


def save_to_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """
    saves dataframe to parquet with zstd compression.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
        coerce_timestamps="us",
    )

    logger.info(f"Saved {len(df):,} rows to {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")


def main() -> None:
    """runs the download."""
    config = SymbolConfig()
    #config = SymbolConfig(max_tickers=5,batch_size=5)

    # make sure input file exists
    if not config.input_csv.exists():
        logger.error(f"Input file not found: {config.input_csv}")
        sys.exit(1)

    # get unique tickers
    tickers_df = extract_unique_tickers(config.input_csv)

    # make output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # download in batches
    tickers = tickers_df["ticker"].tolist()
    
    # limit if needed
    if config.max_tickers is not None:
        original_count = len(tickers)
        tickers = tickers[: config.max_tickers]
        logger.info(f"Limiting download to {config.max_tickers} tickers (out of {original_count} total)")
    
    all_data: list[pd.DataFrame] = []

    logger.info(f"Starting download for {len(tickers)} unique tickers...")
    if config.period:
        logger.info(f"Period: {config.period}")
    elif config.start_date:
        logger.info(f"Date range: {config.start_date} to {config.end_date or 'today'}")
    else:
        logger.info("Period: max (all available historical data)")

    for i in range(0, len(tickers), config.batch_size):
        batch = tickers[i : i + config.batch_size]
        logger.info(f"\nProcessing batch {i // config.batch_size + 1} ({len(batch)} tickers)...")

        batch_df = download_batch(batch, config, start_idx=i)
        
        if batch_df is not None and not batch_df.empty:
            all_data.append(batch_df)
        else:
            logger.warning(f"Batch {i // config.batch_size + 1} produced no data")

    # combine everything
    if not all_data:
        logger.error("No data downloaded!")
        sys.exit(1)

    logger.info("\nCombining all batches...")
    combined_df = pd.concat(all_data, ignore_index=True)

    # save combined file
    combined_path = config.output_dir / "all_stocks_historical.parquet"
    save_to_parquet(combined_df, combined_path)

    # save metadata with actual info from data
    if combined_df.empty:
        raise ValueError("No data to save metadata for!")
    
    successful_tickers = combined_df["ticker"].nunique()
    actual_start_date = combined_df["date"].min().strftime("%Y-%m-%d")
    actual_end_date = combined_df["date"].max().strftime("%Y-%m-%d")
    
    # count which ones actually worked
    tickers_with_data = set(combined_df["ticker"].unique())
    requested_tickers = set(tickers)
    failed_tickers = requested_tickers - tickers_with_data
    
    # figure out what was requested
    if config.period:
        requested_range = f"period={config.period}"
    elif config.start_date:
        requested_range = f"{config.start_date} to {config.end_date or 'today'}"
    else:
        requested_range = "period=max (all available)"
    
    metadata = {
        "total_tickers_requested": len(tickers),
        "successful_downloads": successful_tickers,
        "failed_downloads": len(failed_tickers),
        "requested_range": requested_range,
        "actual_date_range": f"{actual_start_date} to {actual_end_date}",
        "actual_start_date": actual_start_date,
        "actual_end_date": actual_end_date,
        "total_rows": len(combined_df),
        "download_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # add failed tickers list if any (cap at 50)
    if failed_tickers:
        metadata["failed_ticker_list"] = sorted(list(failed_tickers))[:50]
        if len(failed_tickers) > 50:
            metadata["failed_ticker_count"] = len(failed_tickers)
            metadata["note"] = f"Only first 50 failed tickers listed. Total failed: {len(failed_tickers)}"

    metadata_path = config.output_dir / "download_metadata.json"
    import json

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nDownload complete!")
    logger.info(f"Successfully downloaded: {metadata['successful_downloads']}/{metadata['total_tickers_requested']} tickers")
    logger.info(f"Actual date range: {metadata['actual_start_date']} to {metadata['actual_end_date']}")
    logger.info(f"Total rows: {metadata['total_rows']:,}")
    logger.info(f"Combined data saved to: {combined_path}")


if __name__ == "__main__":
    main()

