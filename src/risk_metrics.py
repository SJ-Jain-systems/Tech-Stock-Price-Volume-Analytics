"""Risk metric helpers for returns, volatility, drawdowns, and tail risk."""

from __future__ import annotations

import numpy as np
import pandas as pd

ArrayLike = pd.Series | pd.Index | np.ndarray | list[float] | tuple[float, ...]


def _clean_numeric_series(values: ArrayLike) -> pd.Series:
    """Convert input values to a numeric Series and drop missing observations."""
    return (
        pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    )


def annualized_return(log_returns: ArrayLike, trading_days: int = 252) -> float:
    """Compute annualized return from periodic log returns.

    Parameters
    ----------
    log_returns:
        Periodic log-return observations. Missing and infinite values are
        ignored.
    trading_days:
        Number of return periods in one year.

    Returns
    -------
    float
        Mean log return multiplied by ``trading_days``. Returns ``nan`` when no
        valid observations are available.
    """
    clean_returns = _clean_numeric_series(log_returns)
    if clean_returns.empty:
        return float("nan")
    return float(clean_returns.mean() * trading_days)


def annualized_volatility(log_returns: ArrayLike, trading_days: int = 252) -> float:
    """Compute annualized sample volatility from periodic log returns.

    Missing and infinite values are ignored. At least two valid observations are
    required because sample standard deviation uses ``ddof=1``.
    """
    clean_returns = _clean_numeric_series(log_returns)
    if clean_returns.size < 2:
        return float("nan")
    return float(clean_returns.std(ddof=1) * np.sqrt(trading_days))


def sharpe_like_ratio(
    log_returns: ArrayLike,
    trading_days: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute a Sharpe-like annual excess-return-to-volatility ratio.

    Parameters
    ----------
    log_returns:
        Periodic log-return observations. Missing and infinite values are
        ignored.
    trading_days:
        Number of return periods in one year.
    risk_free_rate:
        Annual risk-free rate expressed in the same return units as the
        annualized log return.

    Returns
    -------
    float
        ``(annualized_return - risk_free_rate) / annualized_volatility``.
        Returns ``nan`` when volatility is missing, zero, or non-finite.
    """
    ann_return = annualized_return(log_returns, trading_days=trading_days)
    ann_volatility = annualized_volatility(log_returns, trading_days=trading_days)

    if not np.isfinite(ann_volatility) or ann_volatility <= 0:
        return float("nan")
    return float((ann_return - risk_free_rate) / ann_volatility)


def max_drawdown(price_series: ArrayLike) -> float:
    """Compute the maximum drawdown of a price series.

    Parameters
    ----------
    price_series:
        Ordered price observations. Missing, infinite, and non-positive prices
        are ignored before computing the running peak path.

    Returns
    -------
    float
        The most negative percentage drawdown, expressed as ``price / running_peak
        - 1``. Returns ``nan`` when no valid positive prices are available.
    """
    prices = _clean_numeric_series(price_series)
    prices = prices[prices > 0]
    if prices.empty:
        return float("nan")

    running_peak = prices.cummax()
    drawdowns = (prices / running_peak) - 1.0
    return float(drawdowns.min())


def value_at_risk(returns: ArrayLike, alpha: float = 0.05) -> float:
    """Compute historical lower-tail Value at Risk (VaR).

    Parameters
    ----------
    returns:
        Periodic simple or log returns. Missing and infinite values are ignored.
    alpha:
        Lower-tail probability between 0 and 1.

    Returns
    -------
    float
        The empirical ``alpha`` quantile of returns. For return series, this is
        typically a negative number during loss scenarios.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    clean_returns = _clean_numeric_series(returns)
    if clean_returns.empty:
        return float("nan")
    return float(clean_returns.quantile(alpha))


def conditional_value_at_risk(returns: ArrayLike, alpha: float = 0.05) -> float:
    """Compute historical Conditional Value at Risk (CVaR).

    CVaR is calculated as the mean of returns at or below the historical VaR
    threshold. Missing and infinite values are ignored.
    """
    clean_returns = _clean_numeric_series(returns)
    if clean_returns.empty:
        return float("nan")

    var_threshold = value_at_risk(clean_returns, alpha=alpha)
    tail_returns = clean_returns[clean_returns <= var_threshold]
    if tail_returns.empty:
        return float("nan")
    return float(tail_returns.mean())


def downside_deviation(returns: ArrayLike, threshold: float = 0.0) -> float:
    """Compute downside deviation below a minimum acceptable return threshold.

    Parameters
    ----------
    returns:
        Periodic return observations. Missing and infinite values are ignored.
    threshold:
        Minimum acceptable return for the downside calculation.

    Returns
    -------
    float
        Square root of the average squared shortfall ``min(return - threshold,
        0)``. Returns ``nan`` when no valid observations are available.
    """
    clean_returns = _clean_numeric_series(returns)
    if clean_returns.empty:
        return float("nan")

    shortfalls = np.minimum(clean_returns - threshold, 0.0)
    return float(np.sqrt(np.mean(np.square(shortfalls))))


def bad_day_rate(returns: ArrayLike, threshold: float = -0.02) -> float:
    """Compute the fraction of valid return observations at or below a threshold.

    Missing and infinite values are ignored. Returns ``nan`` when no valid
    observations are available.
    """
    clean_returns = _clean_numeric_series(returns)
    if clean_returns.empty:
        return float("nan")
    return float((clean_returns <= threshold).mean())
