"""Export processed rolling stock features with drawdown metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DUCKDB_PATH, PROCESSED_FEATURE_PATH
from src.sql_utils import get_connection

EXPORT_QUERY = """
SELECT
    rf.*,
    dd.running_peak,
    dd.drawdown
FROM rolling_features AS rf
LEFT JOIN drawdowns AS dd
    ON rf.symbol = dd.symbol
   AND rf.date = dd.date
ORDER BY rf.symbol, rf.date
"""


def main() -> None:
    """Write the processed feature dataset to data/processed."""

    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB database not found at {DUCKDB_PATH}. Run "
            "scripts/01_build_database.py and scripts/02_run_sql_pipeline.py first."
        )

    PROCESSED_FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = get_connection(DUCKDB_PATH)
    try:
        features = connection.execute(EXPORT_QUERY).df()
        features.to_csv(PROCESSED_FEATURE_PATH, index=False)
    finally:
        connection.close()

    row_count, column_count = features.shape
    print("Exported processed feature dataset")
    print(f"  shape: {row_count:,} rows x {column_count:,} columns")
    print(f"  saved_path: {PROCESSED_FEATURE_PATH}")


if __name__ == "__main__":
    main()
