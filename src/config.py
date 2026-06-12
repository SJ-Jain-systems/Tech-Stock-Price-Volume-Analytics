"""Project-wide configuration constants.

This module centralizes filesystem locations and domain constants used by the
analytics pipeline. Paths are expressed as :class:`pathlib.Path` objects so they
can be passed directly to pandas, DuckDB, and standard-library file APIs.
"""

from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
DATABASE_DIR: Path = DATA_DIR / "database"
SQL_DIR: Path = ROOT_DIR / "sql"
FIGURES_DIR: Path = ROOT_DIR / "reports" / "figures"
MODELS_DIR: Path = ROOT_DIR / "models"

RAW_CSV_PATH: Path = RAW_DATA_DIR / "tech_stocks.csv"
DUCKDB_PATH: Path = DATABASE_DIR / "tech_stocks.duckdb"
PROCESSED_FEATURE_PATH: Path = PROCESSED_DATA_DIR / "tech_stock_features.csv"

TRADING_DAYS_PER_YEAR: int = 252
STOCK_SYMBOLS: tuple[str, ...] = (
    "AAPL",
    "AMZN",
    "FB",
    "GOOG",
    "MSFT",
    "NFLX",
    "NVDA",
)

# Backward-compatible aliases for any existing notebooks/modules that imported
# the previous names before the project configuration was standardized.
PROJECT_ROOT: Path = ROOT_DIR
TECH_SYMBOLS: tuple[str, ...] = STOCK_SYMBOLS
RAW_CSV_NAME: str = RAW_CSV_PATH.name
