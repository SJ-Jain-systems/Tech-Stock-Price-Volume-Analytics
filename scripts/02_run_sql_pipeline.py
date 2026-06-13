"""Run the ordered DuckDB SQL feature-engineering pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DUCKDB_PATH, SQL_DIR
from src.sql_utils import execute_sql_file, get_connection

ROW_COUNT_OBJECTS: tuple[str, ...] = (
    "clean_prices",
    "daily_returns",
    "rolling_features",
    "drawdowns",
    "stock_level_risk_summary",
    "annual_stock_metrics",
    "portfolio_returns_wide",
)


def main() -> None:
    """Execute every versioned SQL script and print core output row counts."""

    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB database not found at {DUCKDB_PATH}. Run "
            "scripts/01_build_database.py first."
        )

    sql_files = sorted(SQL_DIR.glob("*.sql"))
    if not sql_files:
        raise FileNotFoundError(f"No SQL files found in {SQL_DIR}.")

    connection = get_connection(DUCKDB_PATH)
    try:
        print("Running SQL pipeline:")
        for sql_file in sql_files:
            execute_sql_file(connection, sql_file)
            print(f"  executed {sql_file.relative_to(ROOT_DIR)}")

        print("\nPipeline row counts:")
        for object_name in ROW_COUNT_OBJECTS:
            row_count = connection.execute(
                f"SELECT COUNT(*) FROM {object_name}"
            ).fetchone()[0]
            print(f"  {object_name}: {row_count:,}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
