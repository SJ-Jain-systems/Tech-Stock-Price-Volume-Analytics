"""Reusable Bayesian models for daily equity returns.

The helpers in this module are intentionally small and notebook-friendly: data
preparation is separated from model construction, sampling, and posterior
summaries so that research notebooks can reuse the same PyMC model while keeping
intermediate objects easy to inspect.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

REQUIRED_RETURN_COLUMNS = {"symbol", "date", "log_return"}


def _validate_required_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    """Raise a clear error if a DataFrame does not include required columns."""
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _coerce_symbol_filter(symbols: str | Iterable[str] | None) -> list[str] | None:
    """Normalize optional symbol filters while treating one string as one symbol."""
    if symbols is None:
        return None
    if isinstance(symbols, str):
        return [symbols]
    return list(symbols)


def _normalize_symbol_lookup(
    symbol_lookup: Mapping[int, str] | Sequence[str],
) -> list[str]:
    """Return symbols ordered by integer stock index.

    The public preparation function returns a dictionary keyed by integer stock
    index, but summary helpers also accept a list/tuple/Index for convenience in
    notebooks.
    """
    if isinstance(symbol_lookup, Mapping):
        return [symbol_lookup[idx] for idx in sorted(symbol_lookup)]
    return list(symbol_lookup)


def _posterior_variable_samples(
    idata: az.InferenceData,
    variable: str,
    n_symbols: int,
) -> np.ndarray:
    """Extract posterior samples as a ``(n_symbols, n_samples)`` NumPy array."""
    if not hasattr(idata, "posterior") or variable not in idata.posterior:
        raise ValueError(f"InferenceData posterior does not contain {variable!r}")

    draws = idata.posterior[variable]
    stacked = draws.stack(sample=("chain", "draw"))

    # Models created by this module use a named ``stock`` dimension. The fallback
    # supports externally-created but similarly shaped InferenceData objects.
    stock_dims = [dim for dim in stacked.dims if dim != "sample"]
    if len(stock_dims) != 1:
        raise ValueError(
            f"Expected {variable!r} to have exactly one stock dimension; "
            f"found dimensions {draws.dims}."
        )

    samples = stacked.transpose(stock_dims[0], "sample").values
    if samples.shape[0] != n_symbols:
        raise ValueError(
            f"{variable!r} has {samples.shape[0]} stocks, but symbol_lookup "
            f"contains {n_symbols} symbols."
        )
    return np.asarray(samples, dtype="float64")


def prepare_returns_for_bayesian_model(
    df: pd.DataFrame,
    symbols: str | Iterable[str] | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, str], pd.DataFrame]:
    """Clean and encode daily log returns for Bayesian modeling.

    Parameters
    ----------
    df:
        Daily returns table with ``symbol``, ``date``, and ``log_return``
        columns.
    symbols:
        Optional symbol or iterable of symbols to keep.
    start_date, end_date:
        Optional inclusive date bounds.

    Returns
    -------
    tuple
        ``(returns, stock_idx, symbol_lookup, modeling_df)`` where ``returns`` is
        a float array of log returns, ``stock_idx`` is an integer array mapping
        each row to a stock, ``symbol_lookup`` maps stock index to symbol, and
        ``modeling_df`` is the filtered/sorted DataFrame used for modeling.
    """
    _validate_required_columns(df, REQUIRED_RETURN_COLUMNS)

    modeling_df = df.copy()
    modeling_df["date"] = pd.to_datetime(modeling_df["date"])

    symbol_filter = _coerce_symbol_filter(symbols)
    if symbol_filter is not None:
        modeling_df = modeling_df[modeling_df["symbol"].isin(symbol_filter)]

    if start_date is not None:
        modeling_df = modeling_df[modeling_df["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        modeling_df = modeling_df[modeling_df["date"] <= pd.to_datetime(end_date)]

    modeling_df["log_return"] = pd.to_numeric(
        modeling_df["log_return"], errors="coerce"
    )
    modeling_df = modeling_df.replace([np.inf, -np.inf], np.nan)
    modeling_df = modeling_df.dropna(subset=["symbol", "date", "log_return"])
    modeling_df = modeling_df.sort_values(["symbol", "date"], kind="mergesort")
    modeling_df = modeling_df.reset_index(drop=True)

    if modeling_df.empty:
        raise ValueError("No valid log_return observations remain after filtering.")

    stock_codes, unique_symbols = pd.factorize(modeling_df["symbol"], sort=True)
    modeling_df["stock_idx"] = stock_codes.astype("int64")

    returns = modeling_df["log_return"].to_numpy(dtype="float64")
    stock_idx = modeling_df["stock_idx"].to_numpy(dtype="int64")
    symbol_lookup = {idx: symbol for idx, symbol in enumerate(unique_symbols.tolist())}

    return returns, stock_idx, symbol_lookup, modeling_df


def build_hierarchical_student_t_return_model(
    returns: np.ndarray | Sequence[float],
    stock_idx: np.ndarray | Sequence[int],
    n_stocks: int,
) -> pm.Model:
    """Build a hierarchical Student-t model for daily stock log returns.

    The hierarchical prior partially pools stock-level expected returns toward a
    shared group mean. This is useful when individual stocks have noisy daily
    returns: each stock can differ, but extreme estimates are regularized toward
    the cross-sectional average.

    A Student-t likelihood is used because financial returns commonly have fat
    tails and occasional outliers. The degrees-of-freedom parameter is estimated
    with ``nu > 2`` so the likelihood has finite variance while still allowing
    heavier tails than a Gaussian model.
    """
    returns_array = np.asarray(returns, dtype="float64")
    stock_idx_array = np.asarray(stock_idx, dtype="int64")

    if returns_array.ndim != 1:
        raise ValueError("returns must be a one-dimensional array.")
    if stock_idx_array.ndim != 1:
        raise ValueError("stock_idx must be a one-dimensional array.")
    if returns_array.size != stock_idx_array.size:
        raise ValueError("returns and stock_idx must have the same length.")
    if n_stocks < 1:
        raise ValueError("n_stocks must be at least 1.")
    if returns_array.size == 0:
        raise ValueError("At least one return observation is required.")
    if stock_idx_array.min() < 0 or stock_idx_array.max() >= n_stocks:
        raise ValueError("stock_idx values must be in the range [0, n_stocks).")

    coords = {"stock": np.arange(n_stocks), "obs_id": np.arange(returns_array.size)}

    with pm.Model(coords=coords) as model:
        # Daily log returns are usually very close to zero. This weakly
        # informative prior centers the market-wide mean near zero while still
        # allowing economically meaningful daily drift.
        group_mu = pm.Normal("group_mu", mu=0.0, sigma=0.001)

        # Cross-stock variation in expected daily return. The half-normal keeps
        # this scale positive and softly regularizes stock means toward group_mu.
        group_sigma = pm.HalfNormal("group_sigma", sigma=0.02)

        stock_mu_raw = pm.Normal("stock_mu_raw", mu=0.0, sigma=1.0, dims="stock")
        stock_mu = pm.Deterministic(
            "stock_mu",
            group_mu + stock_mu_raw * group_sigma,
            dims="stock",
        )

        # Stock-specific daily volatility. The prior scale is intentionally wide
        # for daily equity returns while still ruling out implausibly huge values.
        stock_sigma = pm.HalfNormal("stock_sigma", sigma=0.03, dims="stock")

        # Estimate tail thickness. Adding two guarantees nu > 2, giving the
        # Student-t likelihood finite variance.
        nu_minus_two = pm.Exponential("nu_minus_two", lam=1 / 10)
        nu = pm.Deterministic("nu", nu_minus_two + 2)

        pm.StudentT(
            "returns",
            nu=nu,
            mu=stock_mu[stock_idx_array],
            sigma=stock_sigma[stock_idx_array],
            observed=returns_array,
            dims="obs_id",
        )

    return model


def sample_model(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
) -> az.InferenceData:
    """Sample a PyMC model and return ArviZ ``InferenceData``."""
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=True,
        )
    return idata


def posterior_stock_summary(
    idata: az.InferenceData,
    symbol_lookup: Mapping[int, str] | Sequence[str],
) -> pd.DataFrame:
    """Summarize posterior ``stock_mu`` and ``stock_sigma`` by symbol.

    Returns posterior mean, standard deviation, and 5%, 50%, and 95% quantiles
    for each stock-level parameter.
    """
    symbols = _normalize_symbol_lookup(symbol_lookup)
    rows: list[dict[str, float | str]] = []

    for variable in ("stock_mu", "stock_sigma"):
        samples = _posterior_variable_samples(idata, variable, len(symbols))
        quantiles = np.quantile(samples, [0.05, 0.50, 0.95], axis=1)

        for idx, symbol in enumerate(symbols):
            rows.append(
                {
                    "symbol": symbol,
                    "parameter": variable,
                    "mean": float(np.mean(samples[idx])),
                    "sd": float(np.std(samples[idx], ddof=1)),
                    "q5": float(quantiles[0, idx]),
                    "q50": float(quantiles[1, idx]),
                    "q95": float(quantiles[2, idx]),
                }
            )

    return pd.DataFrame(rows)


def posterior_probability_positive_return(
    idata: az.InferenceData,
    symbol_lookup: Mapping[int, str] | Sequence[str],
) -> pd.DataFrame:
    """Compute ``Pr(stock_mu > 0)`` for each symbol."""
    symbols = _normalize_symbol_lookup(symbol_lookup)
    stock_mu_samples = _posterior_variable_samples(idata, "stock_mu", len(symbols))
    probabilities = np.mean(stock_mu_samples > 0.0, axis=1)

    return pd.DataFrame(
        {
            "symbol": symbols,
            "probability_positive_return": probabilities.astype("float64"),
        }
    )


def posterior_probability_best_return(
    idata: az.InferenceData,
    symbol_lookup: Mapping[int, str] | Sequence[str],
) -> pd.DataFrame:
    """Compute the posterior probability each stock has the highest ``stock_mu``."""
    symbols = _normalize_symbol_lookup(symbol_lookup)
    stock_mu_samples = _posterior_variable_samples(idata, "stock_mu", len(symbols))
    best_stock_idx = np.argmax(stock_mu_samples, axis=0)
    probabilities = (
        np.bincount(best_stock_idx, minlength=len(symbols)) / best_stock_idx.size
    )

    return pd.DataFrame(
        {
            "symbol": symbols,
            "probability_best_return": probabilities.astype("float64"),
        }
    )


def posterior_predictive_check(
    model: pm.Model,
    idata: az.InferenceData,
) -> az.InferenceData:
    """Run posterior predictive sampling for a fitted return model."""
    with model:
        posterior_predictive = pm.sample_posterior_predictive(
            idata,
            return_inferencedata=True,
            extend_inferencedata=False,
        )
    return posterior_predictive
