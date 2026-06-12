"""Feature engineering utilities for equity price and volume data.

The functions in this module mirror the repository's SQL feature-engineering
pipeline while remaining convenient for notebook- and pandas-based workflows.
Each function returns a new :class:`pandas.DataFrame` and leaves the input frame
unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MOMENTUM_WINDOWS: tuple[int, ...] = (21, 63, 126, 252)
ROLLING_WINDOWS: tuple[int, ...] = (21, 63, 126, 252)
TREND_WINDOWS: tuple[int, ...] = (20, 50, 200)
TRADING_DAYS_PER_YEAR = 252


def _validate_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    """Raise a helpful error when required input columns are missing."""
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _sorted_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Return a stable symbol/date-sorted copy while preserving row index labels."""
    _validate_columns(df, {"symbol", "date"})
    return df.copy().sort_values(["symbol", "date"], kind="mergesort")


def create_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create daily return and liquidity features by symbol.

    Parameters
    ----------
    df:
        Price history with ``symbol``, ``date``, ``adj_close``, and ``volume``
        columns. Prices should be positive for return calculations.

    Returns
    -------
    pandas.DataFrame
        A sorted copy of ``df`` with three additional columns:
        ``simple_return``, ``log_return``, and ``dollar_volume``.

    Notes
    -----
    Missing prices, non-positive prices, and missing prior prices produce
    missing return values. ``dollar_volume`` is missing whenever either price or
    volume is missing.
    """
    _validate_columns(df, {"symbol", "date", "adj_close", "volume"})
    result = _sorted_copy(df)

    current_price = pd.to_numeric(result["adj_close"], errors="coerce")
    volume = pd.to_numeric(result["volume"], errors="coerce")
    previous_price = current_price.groupby(result["symbol"], sort=False).shift(1)

    valid_price_pair = (current_price > 0) & (previous_price > 0)
    price_ratio = current_price / previous_price

    result["simple_return"] = np.where(valid_price_pair, price_ratio - 1.0, np.nan)
    result["log_return"] = np.where(valid_price_pair, np.log(price_ratio), np.nan)
    result["dollar_volume"] = current_price * volume

    return result


def create_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-horizon price momentum features by symbol.

    Parameters
    ----------
    df:
        DataFrame containing ``symbol``, ``date``, and ``adj_close`` columns.

    Returns
    -------
    pandas.DataFrame
        A sorted copy of ``df`` with ``return_21d``, ``return_63d``,
        ``return_126d``, and ``return_252d`` columns. Each feature is computed
        as ``adj_close / lagged_adj_close - 1`` for that lookback window.

    Notes
    -----
    Momentum is left missing until a valid, positive lagged price exists. This
    avoids treating gaps or invalid prices as zero-return observations.
    """
    _validate_columns(df, {"symbol", "date", "adj_close"})
    result = _sorted_copy(df)
    current_price = pd.to_numeric(result["adj_close"], errors="coerce")

    for window in MOMENTUM_WINDOWS:
        lagged_price = current_price.groupby(result["symbol"], sort=False).shift(window)
        valid_price_pair = (current_price > 0) & (lagged_price > 0)
        result[f"return_{window}d"] = np.where(
            valid_price_pair,
            (current_price / lagged_price) - 1.0,
            np.nan,
        )

    return result


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling return mean and volatility features by symbol.

    Parameters
    ----------
    df:
        DataFrame containing ``symbol``, ``date``, and ``simple_return``. If
        ``simple_return`` is absent but ``adj_close`` and ``volume`` are present,
        daily return features are created first.

    Returns
    -------
    pandas.DataFrame
        A sorted copy of ``df`` with rolling mean, rolling volatility, and
        annualized rolling volatility columns for 21-, 63-, 126-, and 252-day
        windows.

    Notes
    -----
    Rolling features require a complete window of non-missing returns via
    ``min_periods=window``. Sample standard deviation (``ddof=1``), matching SQL
    ``STDDEV_SAMP``, is used for volatility.
    """
    if "simple_return" not in df.columns:
        _validate_columns(df, {"symbol", "date", "adj_close", "volume"})
        result = create_return_features(df)
    else:
        _validate_columns(df, {"symbol", "date", "simple_return"})
        result = _sorted_copy(df)

    returns = pd.to_numeric(result["simple_return"], errors="coerce")
    grouped_returns = returns.groupby(result["symbol"], sort=False)

    for window in ROLLING_WINDOWS:
        rolling = grouped_returns.rolling(window=window, min_periods=window)
        result[f"rolling_mean_{window}d"] = rolling.mean().reset_index(
            level=0, drop=True
        )
        result[f"rolling_vol_{window}d"] = rolling.std(ddof=1).reset_index(
            level=0, drop=True
        )
        result[f"rolling_ann_vol_{window}d"] = result[
            f"rolling_vol_{window}d"
        ] * np.sqrt(TRADING_DAYS_PER_YEAR)

    return result


def create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving-average trend features by symbol.

    Parameters
    ----------
    df:
        DataFrame containing ``symbol``, ``date``, and ``adj_close`` columns.

    Returns
    -------
    pandas.DataFrame
        A sorted copy of ``df`` with ``ma_20``, ``ma_50``, ``ma_200`` and the
        corresponding nullable Boolean ``above_ma_20``, ``above_ma_50``, and
        ``above_ma_200`` flags.

    Notes
    -----
    Moving averages require complete windows of non-missing prices. Above-MA
    flags remain missing while the corresponding moving average is unavailable.
    """
    _validate_columns(df, {"symbol", "date", "adj_close"})
    result = _sorted_copy(df)
    prices = pd.to_numeric(result["adj_close"], errors="coerce")
    grouped_prices = prices.groupby(result["symbol"], sort=False)

    for window in TREND_WINDOWS:
        ma_column = f"ma_{window}"
        flag_column = f"above_ma_{window}"
        rolling = grouped_prices.rolling(window=window, min_periods=window)
        result[ma_column] = rolling.mean().reset_index(level=0, drop=True)
        result[flag_column] = (prices > result[ma_column]).where(
            result[ma_column].notna()
        )
        result[flag_column] = result[flag_column].astype("boolean")

    return result
