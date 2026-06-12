"""Portfolio analytics utilities for traditional and Bayesian allocation workflows.

The functions in this module intentionally use long-only portfolios.  Long-only
weights are easier to interpret in a GitHub research project and avoid the
leverage and borrow assumptions that come with short selling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DATE_COLUMNS = {"date", "datetime", "timestamp"}
DEFAULT_TRADING_DAYS = 252

ArrayLike = pd.Series | pd.Index | np.ndarray | list[float] | tuple[float, ...]


def _asset_return_frame(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean numeric returns frame with non-asset date columns removed."""
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame")

    asset_columns = [
        column
        for column in returns_df.columns
        if str(column).lower() not in DATE_COLUMNS
    ]
    if not asset_columns:
        raise ValueError("returns_df must contain at least one asset return column")

    returns = returns_df.loc[:, asset_columns].apply(pd.to_numeric, errors="coerce")
    returns = returns.replace([np.inf, -np.inf], np.nan)
    if returns.shape[1] == 0:
        raise ValueError("returns_df must contain at least one numeric asset column")
    return returns


def _asset_names_from_lookup(
    symbol_lookup: Mapping[Any, str] | Sequence[str],
) -> list[str]:
    """Extract asset symbols from a mapping or sequence in deterministic order."""
    if isinstance(symbol_lookup, Mapping):
        try:
            keys = sorted(symbol_lookup)
        except TypeError:
            keys = list(symbol_lookup)
        return [str(symbol_lookup[key]) for key in keys]
    return [str(symbol) for symbol in symbol_lookup]


def _as_numeric_vector(values: ArrayLike, name: str) -> np.ndarray:
    """Convert one-dimensional numeric input to a finite NumPy vector."""
    vector = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype="float64")
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return vector


def _coerce_weights(weights: ArrayLike, n_assets: int | None = None) -> np.ndarray:
    """Validate and normalize a long-only weight vector."""
    weights_array = _as_numeric_vector(weights, "weights")
    if n_assets is not None and weights_array.size != n_assets:
        raise ValueError(
            f"weights length ({weights_array.size}) must match number of assets "
            f"({n_assets})"
        )
    if np.any(weights_array < -1e-10):
        raise ValueError("weights must be long-only; negative weights are not allowed")

    weights_array = np.clip(weights_array, 0.0, None)
    total_weight = weights_array.sum()
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("weights must have a positive finite sum")
    return weights_array / total_weight


def _coerce_mean_covariance(
    mean_returns: ArrayLike,
    covariance_matrix: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    """Validate mean and covariance inputs and return NumPy arrays plus labels."""
    labels = list(mean_returns.index) if isinstance(mean_returns, pd.Series) else None
    mean_array = _as_numeric_vector(mean_returns, "mean_returns")

    covariance_array = np.asarray(covariance_matrix, dtype="float64")
    if (
        covariance_array.ndim != 2
        or covariance_array.shape[0] != covariance_array.shape[1]
    ):
        raise ValueError("covariance_matrix must be a square matrix")
    if covariance_array.shape[0] != mean_array.size:
        raise ValueError(
            "mean_returns length must match covariance_matrix dimensions; "
            f"got {mean_array.size} means and "
            f"{covariance_array.shape[0]} covariance rows"
        )
    if not np.all(np.isfinite(covariance_array)):
        raise ValueError("covariance_matrix must contain only finite numeric values")

    covariance_array = (covariance_array + covariance_array.T) / 2.0
    return mean_array, covariance_array, labels


def _coerce_covariance(
    covariance_matrix: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
) -> tuple[np.ndarray, list[str] | None]:
    """Validate a covariance matrix and return a symmetric NumPy array plus labels."""
    labels = (
        list(covariance_matrix.index)
        if isinstance(covariance_matrix, pd.DataFrame)
        else None
    )
    covariance_array = np.asarray(covariance_matrix, dtype="float64")
    if (
        covariance_array.ndim != 2
        or covariance_array.shape[0] != covariance_array.shape[1]
    ):
        raise ValueError("covariance_matrix must be a square matrix")
    if covariance_array.shape[0] == 0:
        raise ValueError("covariance_matrix must contain at least one asset")
    if not np.all(np.isfinite(covariance_array)):
        raise ValueError("covariance_matrix must contain only finite numeric values")
    return (covariance_array + covariance_array.T) / 2.0, labels


def _weights_output(
    weights: np.ndarray,
    labels: list[str] | None,
) -> pd.Series | np.ndarray:
    """Return weights as a labeled Series when labels are available."""
    if labels is None:
        return weights
    return pd.Series(weights, index=labels, name="weight")


def _clean_optimized_weights(weights: np.ndarray) -> np.ndarray:
    """Remove numerical optimizer noise while preserving the long-only simplex."""
    cleaned = np.where(weights < 1e-10, 0.0, weights)
    total = cleaned.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("optimizer returned invalid weights")
    return cleaned / total


def compute_equal_weight_portfolio(returns_df: pd.DataFrame) -> pd.Series:
    """Compute equal-weight portfolio daily returns from a wide return table.

    Parameters
    ----------
    returns_df:
        Wide return DataFrame where each asset has its own column.  A date-like
        column named ``date``, ``datetime``, or ``timestamp`` is ignored if
        present.  Missing asset returns are skipped for that date.

    Returns
    -------
    pandas.Series
        Daily equal-weight portfolio returns indexed like ``returns_df``.  The
        series is named ``equal_weight_return``.
    """
    returns = _asset_return_frame(returns_df)
    portfolio_returns = returns.mean(axis=1, skipna=True)
    portfolio_returns.name = "equal_weight_return"
    return portfolio_returns


def estimate_mean_covariance(
    returns_df: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate sample mean returns and the sample covariance matrix.

    Missing and infinite observations are ignored using pandas' standard
    pairwise covariance behavior.  Returns are assumed to be daily unless the
    caller documents otherwise.
    """
    returns = _asset_return_frame(returns_df)
    mean_returns = returns.mean(skipna=True)
    covariance_matrix = returns.cov()

    if mean_returns.isna().all():
        raise ValueError("No valid return observations are available to estimate means")
    if covariance_matrix.isna().all().all():
        raise ValueError(
            "No valid return observations are available to estimate covariance"
        )

    return mean_returns, covariance_matrix


def random_long_only_weights(
    n_assets: int,
    n_portfolios: int = 10000,
    seed: int | None = 42,
) -> np.ndarray:
    """Generate random long-only portfolio weights that each sum to one.

    Weights are sampled from a symmetric Dirichlet distribution, which naturally
    enforces non-negative weights and a total portfolio weight of one.
    """
    if not isinstance(n_assets, int) or n_assets <= 0:
        raise ValueError("n_assets must be a positive integer")
    if not isinstance(n_portfolios, int) or n_portfolios <= 0:
        raise ValueError("n_portfolios must be a positive integer")

    rng = np.random.default_rng(seed)
    return rng.dirichlet(alpha=np.ones(n_assets), size=n_portfolios)


def portfolio_performance(
    weights: ArrayLike,
    mean_returns: ArrayLike,
    covariance_matrix: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    trading_days: int = DEFAULT_TRADING_DAYS,
) -> dict[str, float]:
    """Compute annualized return, volatility, and a Sharpe-like ratio.

    The Sharpe-like ratio uses a zero risk-free rate:
    ``annualized_return / annualized_volatility``.
    """
    if trading_days <= 0:
        raise ValueError("trading_days must be positive")

    mean_array, covariance_array, _ = _coerce_mean_covariance(
        mean_returns, covariance_matrix
    )
    weights_array = _coerce_weights(weights, n_assets=mean_array.size)

    daily_return = float(weights_array @ mean_array)
    daily_variance = float(weights_array @ covariance_array @ weights_array)
    daily_variance = max(daily_variance, 0.0)

    annualized_return = daily_return * trading_days
    annualized_volatility = float(np.sqrt(daily_variance * trading_days))
    sharpe_like_ratio = (
        annualized_return / annualized_volatility
        if annualized_volatility > 0 and np.isfinite(annualized_volatility)
        else float("nan")
    )

    return {
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_like_ratio": float(sharpe_like_ratio),
    }


def optimize_max_sharpe_long_only(
    mean_returns: ArrayLike,
    covariance_matrix: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
) -> tuple[pd.Series | np.ndarray, dict[str, float]]:
    """Optimize a long-only portfolio for the highest Sharpe-like ratio.

    The optimization uses SciPy's SLSQP solver with these constraints:
    weights are non-negative and sum to one.  The risk-free rate is assumed to be
    zero to match :func:`portfolio_performance`.
    """
    mean_array, covariance_array, labels = _coerce_mean_covariance(
        mean_returns, covariance_matrix
    )
    n_assets = mean_array.size
    initial_weights = np.full(n_assets, 1.0 / n_assets)
    bounds = [(0.0, 1.0)] * n_assets
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}

    def negative_sharpe(weights: np.ndarray) -> float:
        daily_return = float(weights @ mean_array)
        daily_variance = max(float(weights @ covariance_array @ weights), 0.0)
        daily_volatility = float(np.sqrt(daily_variance))
        if daily_volatility <= 0 or not np.isfinite(daily_volatility):
            return 1e12
        return -(daily_return / daily_volatility)

    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise ValueError(f"Max Sharpe optimization failed: {result.message}")

    weights_array = _clean_optimized_weights(result.x)
    performance = portfolio_performance(weights_array, mean_array, covariance_array)
    return _weights_output(weights_array, labels), performance


def optimize_min_volatility_long_only(
    covariance_matrix: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
) -> tuple[pd.Series | np.ndarray, dict[str, float]]:
    """Optimize a long-only portfolio for minimum annualized volatility.

    Because this function receives only a covariance matrix, the returned
    performance dictionary reports volatility and uses ``nan`` for return and
    Sharpe-like ratio.
    """
    covariance_array, labels = _coerce_covariance(covariance_matrix)
    n_assets = covariance_array.shape[0]
    initial_weights = np.full(n_assets, 1.0 / n_assets)
    bounds = [(0.0, 1.0)] * n_assets
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}

    def variance(weights: np.ndarray) -> float:
        return max(float(weights @ covariance_array @ weights), 0.0)

    result = minimize(
        variance,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise ValueError(f"Minimum volatility optimization failed: {result.message}")

    weights_array = _clean_optimized_weights(result.x)
    annualized_volatility = float(
        np.sqrt(variance(weights_array) * DEFAULT_TRADING_DAYS)
    )
    performance = {
        "annualized_return": float("nan"),
        "annualized_volatility": annualized_volatility,
        "sharpe_like_ratio": float("nan"),
    }
    return _weights_output(weights_array, labels), performance


def historical_var_cvar(
    portfolio_returns: ArrayLike,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute historical lower-tail Value at Risk and Conditional VaR.

    VaR is the empirical ``alpha`` quantile of portfolio returns.  CVaR is the
    average of returns at or below that VaR threshold.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    clean_returns = (
        pd.to_numeric(pd.Series(portfolio_returns), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if clean_returns.empty:
        return float("nan"), float("nan")

    var = float(clean_returns.quantile(alpha))
    tail_returns = clean_returns[clean_returns <= var]
    cvar = float(tail_returns.mean()) if not tail_returns.empty else float("nan")
    return var, cvar


def _posterior_draw_frame(
    draws: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    symbols: list[str],
    name: str,
) -> pd.DataFrame:
    """Coerce posterior draws into a draw-by-symbol DataFrame."""
    if isinstance(draws, pd.DataFrame):
        frame = draws.copy()
        missing = [symbol for symbol in symbols if symbol not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing symbol columns: {missing}")
        frame = frame.loc[:, symbols]
    else:
        array = np.asarray(draws, dtype="float64")
        if array.ndim != 2:
            raise ValueError(
                f"{name} must be a two-dimensional draw array or DataFrame"
            )
        if array.shape[1] == len(symbols):
            oriented = array
        elif array.shape[0] == len(symbols):
            oriented = array.T
        else:
            raise ValueError(
                f"{name} shape {array.shape} is incompatible with "
                f"{len(symbols)} symbols"
            )
        frame = pd.DataFrame(oriented, columns=symbols)

    frame = frame.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    if frame.isna().any().any():
        raise ValueError(f"{name} contains missing or non-finite values")
    return frame.reset_index(drop=True)


def _aligned_correlation_matrix(
    corr_matrix: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    symbols: list[str],
) -> np.ndarray:
    """Return a finite correlation matrix aligned to the supplied symbols."""
    if isinstance(corr_matrix, pd.DataFrame):
        missing = [
            symbol
            for symbol in symbols
            if symbol not in corr_matrix.index or symbol not in corr_matrix.columns
        ]
        if missing:
            raise ValueError(f"corr_matrix is missing symbols: {missing}")
        corr_array = corr_matrix.loc[symbols, symbols].to_numpy(dtype="float64")
    else:
        corr_array = np.asarray(corr_matrix, dtype="float64")

    if corr_array.shape != (len(symbols), len(symbols)):
        raise ValueError(
            "corr_matrix dimensions must match the number of symbols; "
            f"got {corr_array.shape} for {len(symbols)} symbols"
        )
    if not np.all(np.isfinite(corr_array)):
        raise ValueError("corr_matrix must contain only finite values")
    return (corr_array + corr_array.T) / 2.0


def bayesian_portfolio_simulation(
    posterior_return_draws: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    posterior_vol_draws: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    corr_matrix: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    symbol_lookup: Mapping[Any, str] | Sequence[str],
) -> pd.DataFrame:
    """Optimize a long-only max-Sharpe portfolio for each posterior draw.

    For every posterior sample, the function builds a covariance matrix as
    ``diag(volatility) @ correlation @ diag(volatility)`` and optimizes
    long-only max-Sharpe weights.  The output includes one row per posterior
    draw, weight columns named ``weight_<symbol>``, and portfolio metric columns.
    """
    symbols = _asset_names_from_lookup(symbol_lookup)
    if not symbols:
        raise ValueError("symbol_lookup must contain at least one symbol")

    return_draws = _posterior_draw_frame(
        posterior_return_draws, symbols, "posterior_return_draws"
    )
    vol_draws = _posterior_draw_frame(
        posterior_vol_draws, symbols, "posterior_vol_draws"
    )
    if len(return_draws) != len(vol_draws):
        raise ValueError(
            "posterior_return_draws and posterior_vol_draws must have the same "
            "number of draws"
        )

    corr_array = _aligned_correlation_matrix(corr_matrix, symbols)
    records: list[dict[str, float | int]] = []

    for draw_idx in range(len(return_draws)):
        mean_vector = return_draws.iloc[draw_idx].to_numpy(dtype="float64")
        vol_vector = vol_draws.iloc[draw_idx].to_numpy(dtype="float64")
        if np.any(vol_vector < 0):
            raise ValueError("posterior_vol_draws must be non-negative")

        covariance_matrix = np.outer(vol_vector, vol_vector) * corr_array
        weights, performance = optimize_max_sharpe_long_only(
            mean_vector, covariance_matrix
        )
        weights_array = np.asarray(weights, dtype="float64")

        record: dict[str, float | int] = {"draw": draw_idx}
        record.update(
            {
                f"weight_{symbol}": float(weight)
                for symbol, weight in zip(symbols, weights_array)
            }
        )
        record.update(performance)
        records.append(record)

    return pd.DataFrame.from_records(records)


def summarize_portfolio_weights(weights_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize posterior or simulated portfolio weights by symbol.

    The function accepts either plain symbol columns or columns prefixed with
    ``weight_``.  It returns mean, median, 5th percentile, and 95th percentile
    for each symbol's weight.
    """
    if not isinstance(weights_df, pd.DataFrame):
        raise TypeError("weights_df must be a pandas DataFrame")

    weight_columns = [
        column
        for column in weights_df.columns
        if str(column).startswith("weight_")
    ]
    if not weight_columns:
        numeric_columns = weights_df.select_dtypes(include=[np.number]).columns.tolist()
        weight_columns = [column for column in numeric_columns if str(column) != "draw"]
    if not weight_columns:
        raise ValueError("weights_df must contain weight columns to summarize")

    weights = weights_df.loc[:, weight_columns].apply(pd.to_numeric, errors="coerce")
    summary = pd.DataFrame(
        {
            "mean": weights.mean(skipna=True),
            "median": weights.median(skipna=True),
            "weight_5pct": weights.quantile(0.05),
            "weight_95pct": weights.quantile(0.95),
        }
    )
    summary.index = [str(column).removeprefix("weight_") for column in summary.index]
    summary.index.name = "symbol"
    return summary
