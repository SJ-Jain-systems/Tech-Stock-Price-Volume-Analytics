"""Build the DuckDB database from the raw technology stock CSV."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DUCKDB_PATH, RAW_CSV_PATH, SQL_DIR
from src.data_loader import load_raw_stock_data
from src.sql_utils import execute_sql_file, get_connection


def main() -> None:
    """Create the raw and cleaned DuckDB tables from the canonical raw CSV."""

    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Expected raw CSV at {RAW_CSV_PATH}. Copy tech_stocks.csv into "
            "data/raw/tech_stocks.csv before running this script."
        )

    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DUCKDB_PATH.exists():
        DUCKDB_PATH.unlink()

    raw_prices = load_raw_stock_data(RAW_CSV_PATH)

    connection = get_connection(DUCKDB_PATH)
    try:
        execute_sql_file(connection, SQL_DIR / "01_create_tables.sql")
        connection.register("raw_prices_df", raw_prices)
        try:
            connection.execute(
                """
                DELETE FROM raw_prices;

                INSERT INTO raw_prices
                SELECT
                    symbol,
                    CAST(date AS DATE) AS date,
                    open,
                    high,
                    low,
                    close,
                    close_adjusted,
                    CAST(volume AS BIGINT) AS volume,
                    split_coefficient
                FROM raw_prices_df;
                """
            )
        finally:
            connection.unregister("raw_prices_df")

        execute_sql_file(connection, SQL_DIR / "02_clean_prices.sql")

        raw_row_count = connection.execute("SELECT COUNT(*) FROM raw_prices").fetchone()[0]
        clean_row_count = connection.execute("SELECT COUNT(*) FROM clean_prices").fetchone()[0]
        symbol_count = connection.execute(
            "SELECT COUNT(DISTINCT symbol) FROM clean_prices"
        ).fetchone()[0]
        min_date, max_date = connection.execute(
            "SELECT MIN(date), MAX(date) FROM clean_prices"
        ).fetchone()

        print("Built DuckDB database")
        print(f"  raw_csv: {RAW_CSV_PATH}")
        print(f"  database: {DUCKDB_PATH}")
        print(f"  raw_prices rows: {raw_row_count:,}")
        print(f"  clean_prices rows: {clean_row_count:,}")
        print(f"  symbols: {symbol_count}")
        print(f"  date range: {min_date} to {max_date}")
        print("\nData quality summary:")
        print(connection.execute("SELECT * FROM data_quality_summary").df().to_string(index=False))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
