"""
check symbol deduplication.

Run this before downloading to verify we dont have dupes
"""

import pandas as pd
from pathlib import Path


def main() -> None:
    
    csv_path = Path("../../data/all_symbols_2025_11_13-1027.csv")

    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df):,}")
    print(f"Unique symbols (with exchange): {df['symbol'].nunique():,}")

    # Extract ticker
    df["ticker"] = df["symbol"].str.split(":").str[-1]
    unique_tickers = df["ticker"].nunique()

    print(f"Unique tickers (no exchange): {unique_tickers:,}")
    print(f"Duplicates removed: {len(df) - unique_tickers:,}")

    # Show examples of duplicates if any
    duplicates = df[df.duplicated(subset=["ticker"], keep=False)]
    if len(duplicates) > 0:
        print(f"\nFound {len(duplicates)} duplicate entries:")
        print(duplicates[["symbol", "exchange", "ticker"]].head(20))
    else:
        print("\nNo duplicates found!")

    # Show exchange distribution
    print("\nExchange distribution:")
    print(df["exchange"].value_counts())


if __name__ == "__main__":
    main()

