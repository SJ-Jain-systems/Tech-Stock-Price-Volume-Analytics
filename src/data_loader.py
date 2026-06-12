"""Data loading helpers for validating and importing raw tech stock prices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame

from src.config import RAW_CSV_PATH

REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "close_adjusted",
    "volume",
    "split_coefficient",
)
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "close_adjusted")
POSITIVE_VALUE_COLUMNS: tuple[str, ...] = (*PRICE_COLUMNS, "volume")


def load_raw_stock_data(path: str | Path = RAW_CSV_PATH) -> DataFrame:
    """Load, clean, and validate the raw technology stock CSV.

    Parameters
    ----------
    path:
        CSV file location. Defaults to ``data/raw/tech_stocks.csv``.

    Returns
    -------
    pandas.DataFrame
        Clean stock-price data sorted by ``symbol`` and ``date``.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing, dates are invalid, or price/volume
        fields contain missing, non-numeric, zero, or negative values.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw stock CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")

    _validate_required_columns(df, csv_path)
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        invalid_count = int(df["date"].isna().sum())
        raise ValueError(
            f"Column 'date' contains {invalid_count} invalid or missing value(s) "
            f"in {csv_path}."
        )

    for column in POSITIVE_VALUE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    _validate_positive_values(df, POSITIVE_VALUE_COLUMNS, csv_path)

    return df.sort_values(["symbol", "date"], kind="mergesort").reset_index(
        drop=True
    )


def summarize_dataset(df: DataFrame) -> dict[str, Any]:
    """Return high-level quality and coverage metrics for a stock dataset.

    Parameters
    ----------
    df:
        Stock-price dataframe to summarize. It must contain ``symbol`` and
        ``date`` columns so date coverage can be computed by ticker.

    Returns
    -------
    dict[str, Any]
        Summary containing row/column counts, symbols, per-symbol date ranges,
        missing values by column, and duplicate row count.
    """

    _validate_summary_columns(df)
    summary_df = df.copy()
    summary_df["date"] = pd.to_datetime(summary_df["date"], errors="coerce")

    date_ranges = (
        summary_df.groupby("symbol", sort=True)["date"]
        .agg(min_date="min", max_date="max")
        .reset_index()
    )

    return {
        "row_count": int(len(summary_df)),
        "column_count": int(len(summary_df.columns)),
        "symbols": sorted(summary_df["symbol"].dropna().unique().tolist()),
        "date_range_by_symbol": {
            row["symbol"]: {"min_date": row["min_date"], "max_date": row["max_date"]}
            for row in date_ranges.to_dict(orient="records")
        },
        "missing_values_by_column": summary_df.isna().sum().astype(int).to_dict(),
        "duplicate_row_count": int(summary_df.duplicated().sum()),
    }


def _validate_required_columns(df: DataFrame, csv_path: Path) -> None:
    """Raise a clear error if any expected raw CSV columns are absent."""

    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(df.columns))
    if missing_columns:
        raise ValueError(
            f"Raw stock CSV at {csv_path} is missing required column(s): "
            f"{', '.join(missing_columns)}."
        )


def _validate_positive_values(
    df: DataFrame, columns: tuple[str, ...], csv_path: Path
) -> None:
    """Validate that numeric columns contain strictly positive values."""

    invalid_counts = {
        column: int((df[column].isna() | (df[column] <= 0)).sum())
        for column in columns
        if (df[column].isna() | (df[column] <= 0)).any()
    }
    if invalid_counts:
        formatted_counts = ", ".join(
            f"{column}={count}" for column, count in invalid_counts.items()
        )
        raise ValueError(
            f"Raw stock CSV at {csv_path} contains missing, non-numeric, zero, or "
            f"negative price/volume value(s): {formatted_counts}."
        )


def _validate_summary_columns(df: DataFrame) -> None:
    """Validate columns needed to summarize stock coverage by symbol and date."""

    missing_columns = sorted({"symbol", "date"}.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "Cannot summarize dataset because required column(s) are missing: "
            f"{', '.join(missing_columns)}."
        )
