import pyarrow.parquet as pq
import pyarrow.csv as pc
from pathlib import Path

PARQUET_PATH = Path("data/historical/all_stocks_historical.parquet")
csv_path = Path("data/tick5_all_stocks_historical.csv")
table = pq.read_table(PARQUET_PATH)
pc.write_csv(table, csv_path)
