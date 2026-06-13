"""Reusable plotting helpers for financial analysis reports.

The functions in this module create clean, static matplotlib/seaborn figures
that render well in notebooks, Quarto reports, and GitHub previews. Each helper
returns the ``(figure, axes)`` objects so callers can further customize or test
plots before display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, TRADING_DAYS_PER_YEAR

DEFAULT_FIGSIZE = (11, 6)
DEFAULT_DPI = 150
DEFAULT_PALETTE = "colorblind"


__all__ = [
    "plot_adjusted_prices",
    "plot_cumulative_returns",
    "plot_rolling_volatility",
    "plot_drawdowns",
    "plot_return_distribution",
    "plot_correlation_heatmap",
    "plot_posterior_intervals",
    "plot_portfolio_weights",
]


def _validate_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    """Raise a clear error if a DataFrame is missing required columns."""
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _copy_with_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted copy with a datetime-like ``date`` column."""
    _validate_columns(df, {"date"})
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["date"]).sort_values("date", kind="mergesort")
    if result.empty:
        raise ValueError("DataFrame contains no valid date values to plot")
    return result


def _maybe_save_figure(fig: plt.Figure, output_path: str | Path | None) -> None:
    """Save ``fig`` when an output path is provided.

    Bare filenames such as ``"prices.png"`` are saved under
    ``reports/figures``. Paths with a parent directory are honored as supplied.
    """
    if output_path is None:
        return

    path = Path(output_path)
    if not path.is_absolute() and path.parent == Path("."):
        path = FIGURES_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight")


def _format_time_axis(ax: plt.Axes) -> None:
    """Apply readable date formatting to an axis."""
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    )


def _apply_common_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply common titles, labels, grid, and despine settings."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    sns.despine(ax=ax)


def _prepare_return_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy frame with ``symbol``, ``date``, and ``simple_return``."""
    _validate_columns(df, {"symbol", "date"})
    result = _copy_with_datetime(df)

    if "simple_return" in result.columns:
        result["simple_return"] = pd.to_numeric(
            result["simple_return"], errors="coerce"
        )
        return result.dropna(subset=["simple_return"])

    _validate_columns(result, {"adj_close"})
    result = result.sort_values(["symbol", "date"], kind="mergesort")
    prices = pd.to_numeric(result["adj_close"], errors="coerce")
    result["simple_return"] = prices.groupby(result["symbol"], sort=False).pct_change()
    return result.dropna(subset=["simple_return"])


def plot_adjusted_prices(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot adjusted closing prices by symbol.

    Parameters
    ----------
    df:
        Long-form price history with ``date``, ``symbol``, and ``adj_close``.
    output_path:
        Optional destination. Bare filenames are saved to ``reports/figures``.
    """
    _validate_columns(df, {"symbol", "date", "adj_close"})
    plot_df = _copy_with_datetime(df)
    plot_df["adj_close"] = pd.to_numeric(plot_df["adj_close"], errors="coerce")
    plot_df = plot_df.dropna(subset=["adj_close"])

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.lineplot(
        data=plot_df,
        x="date",
        y="adj_close",
        hue="symbol",
        palette=DEFAULT_PALETTE,
        linewidth=1.8,
        ax=ax,
    )
    _apply_common_style(
        ax, "Adjusted Close Prices by Symbol", "Date", "Adjusted close price"
    )
    _format_time_axis(ax)
    ax.legend(title="Symbol", frameon=False, ncol=2)
    fig.tight_layout()
    _maybe_save_figure(fig, output_path)
    return fig, ax


def plot_cumulative_returns(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot cumulative returns by symbol.

    Uses ``simple_return`` when available; otherwise returns are derived from
    ``adj_close``. The y-axis is displayed in percentage terms.
    """
    plot_df = _prepare_return_frame(df).sort_values(
        ["symbol", "date"], kind="mergesort"
    )
    plot_df["cumulative_return"] = (1.0 + plot_df["simple_return"]).groupby(
        plot_df["symbol"], sort=False
    ).cumprod() - 1.0

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.lineplot(
        data=plot_df,
        x="date",
        y="cumulative_return",
        hue="symbol",
        palette=DEFAULT_PALETTE,
        linewidth=1.8,
        ax=ax,
    )
    _apply_common_style(ax, "Cumulative Returns by Symbol", "Date", "Cumulative return")
    _format_time_axis(ax)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.legend(title="Symbol", frameon=False, ncol=2)
    fig.tight_layout()
    _maybe_save_figure(fig, output_path)
    return fig, ax


def plot_rolling_volatility(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot annualized rolling volatility by symbol.

    If one or more ``rolling_ann_vol_*d`` columns are present, the longest
    available window is plotted. Otherwise the function computes a 63-trading-day
    annualized rolling volatility from ``simple_return`` or ``adj_close``.
    """
    _validate_columns(df, {"symbol", "date"})
    result = _copy_with_datetime(df)
    volatility_columns = [
        col for col in result.columns if col.startswith("rolling_ann_vol_")
    ]

    if volatility_columns:
        windowed_columns = sorted(
            volatility_columns,
            key=lambda col: int("".join(ch for ch in col if ch.isdigit()) or 0),
        )
        volatility_col = windowed_columns[-1]
        window_label = volatility_col.replace("rolling_ann_vol_", "").upper()
        plot_df = result[["symbol", "date", volatility_col]].copy()
        plot_df["annualized_volatility"] = pd.to_numeric(
            plot_df[volatility_col], errors="coerce"
        )
    else:
        plot_df = _prepare_return_frame(result).sort_values(
            ["symbol", "date"], kind="mergesort"
        )
        window = 63
        window_label = f"{window}D"
        plot_df["annualized_volatility"] = plot_df.groupby("symbol", sort=False)[
            "simple_return"
        ].rolling(window=window, min_periods=window).std(ddof=1).reset_index(
            level=0, drop=True
        ) * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )

    plot_df = plot_df.dropna(subset=["annualized_volatility"])
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.lineplot(
        data=plot_df,
        x="date",
        y="annualized_volatility",
        hue="symbol",
        palette=DEFAULT_PALETTE,
        linewidth=1.8,
        ax=ax,
    )
    _apply_common_style(
        ax,
        f"Annualized Rolling Volatility ({window_label})",
        "Date",
        "Annualized volatility",
    )
    _format_time_axis(ax)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.legend(title="Symbol", frameon=False, ncol=2)
    fig.tight_layout()
    _maybe_save_figure(fig, output_path)
    return fig, ax


def plot_drawdowns(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot drawdowns from each symbol's running wealth peak.

    Uses ``adj_close`` when available. If prices are absent, drawdowns are
    computed from cumulative wealth based on ``simple_return``.
    """
    _validate_columns(df, {"symbol", "date"})
    result = _copy_with_datetime(df).sort_values(["symbol", "date"], kind="mergesort")

    if "adj_close" in result.columns:
        result["wealth_index"] = pd.to_numeric(result["adj_close"], errors="coerce")
    else:
        result = _prepare_return_frame(result).sort_values(
            ["symbol", "date"], kind="mergesort"
        )
        result["wealth_index"] = (
            (1.0 + result["simple_return"])
            .groupby(result["symbol"], sort=False)
            .cumprod()
        )

    running_peak = result.groupby("symbol", sort=False)["wealth_index"].cummax()
    result["drawdown"] = result["wealth_index"] / running_peak - 1.0
    plot_df = result.dropna(subset=["drawdown"])

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.lineplot(
        data=plot_df,
        x="date",
        y="drawdown",
        hue="symbol",
        palette=DEFAULT_PALETTE,
        linewidth=1.6,
        ax=ax,
    )
    _apply_common_style(ax, "Drawdowns by Symbol", "Date", "Drawdown from peak")
    _format_time_axis(ax)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.legend(title="Symbol", frameon=False, ncol=2)
    fig.tight_layout()
    _maybe_save_figure(fig, output_path)
    return fig, ax


def plot_return_distribution(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot daily return distributions by symbol.

    Uses ``simple_return`` when present; otherwise derives returns from
    ``adj_close``. Distributions are shown as density curves for readability.
    """
    plot_df = _prepare_return_frame(df).copy()

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.kdeplot(
        data=plot_df,
        x="simple_return",
        hue="symbol",
        palette=DEFAULT_PALETTE,
        common_norm=False,
        linewidth=1.8,
        ax=ax,
    )
    _apply_common_style(
        ax, "Daily Return Distribution by Symbol", "Daily return", "Density"
    )
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    _maybe_save_figure(fig, output_path)
    return fig, ax


def plot_correlation_heatmap(
    corr_df_or_matrix: pd.DataFrame | np.ndarray,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a correlation heatmap from a square DataFrame or matrix."""
    corr_df = pd.DataFrame(corr_df_or_matrix).copy()
    if corr_df.shape[0] != corr_df.shape[1]:
        raise ValueError("Correlation input must be a square matrix")
    corr_df = corr_df.apply(pd.to_numeric, errors="coerce")

    fig_width = max(7, min(14, corr_df.shape[1] * 0.8 + 3))
    fig_height = max(6, min(12, corr_df.shape[0] * 0.7 + 2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        corr_df,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )
    ax.set_title("Return Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Asset")
    ax.set_ylabel("Asset")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _maybe_save_figure(fig, output_path)
    return fig, ax


def plot_posterior_intervals(
    summary_df: pd.DataFrame,
    value_col: str,
    lower_col: str,
    upper_col: str,
    label_col: str,
    title: str,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot posterior point estimates with lower/upper interval bounds."""
    _validate_columns(summary_df, {value_col, lower_col, upper_col, label_col})
    plot_df = summary_df[[label_col, value_col, lower_col, upper_col]].copy()
    for col in (value_col, lower_col, upper_col):
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col, lower_col, upper_col]).sort_values(
        value_col
    )

    y_positions = np.arange(len(plot_df))
    lower_errors = plot_df[value_col] - plot_df[lower_col]
    upper_errors = plot_df[upper_col] - plot_df[value_col]

    fig_height = max(4.5, min(12, 0.45 * len(plot_df) + 2))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.errorbar(
        plot_df[value_col],
        y_positions,
        xerr=[lower_errors, upper_errors],
        fmt="o",
        color="#1f77b4",
        ecolor="#6baed6",
        elinewidth=2,
        capsize=4,
        markersize=6,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df[label_col].astype(str))
    _apply_common_style(
        ax,
        title,
        value_col.replace("_", " ").title(),
        label_col.replace("_", " ").title(),
    )
    fig.tight_layout()
    _maybe_save_figure(fig, output_path)
    return fig, ax


def _coerce_weights_to_long(weights_df: pd.DataFrame) -> pd.DataFrame:
    """Convert common portfolio-weight layouts into symbol/portfolio/weight rows."""
    if not isinstance(weights_df, pd.DataFrame):
        raise TypeError("weights_df must be a pandas DataFrame")

    df = weights_df.copy()
    lower_columns = {str(col).lower(): col for col in df.columns}
    symbol_col = (
        lower_columns.get("symbol")
        or lower_columns.get("asset")
        or lower_columns.get("ticker")
    )
    weight_col = lower_columns.get("weight")
    portfolio_col = (
        lower_columns.get("portfolio")
        or lower_columns.get("method")
        or lower_columns.get("strategy")
    )

    if symbol_col is not None and weight_col is not None:
        columns = [symbol_col, weight_col]
        if portfolio_col is not None:
            columns.append(portfolio_col)
        long_df = df[columns].rename(
            columns={
                symbol_col: "symbol",
                weight_col: "weight",
                portfolio_col: "portfolio",
            }
        )
        if "portfolio" not in long_df.columns:
            long_df["portfolio"] = "Portfolio"
        return long_df

    if symbol_col is not None:
        value_columns = [col for col in df.columns if col != symbol_col]
        return df.melt(
            id_vars=symbol_col,
            value_vars=value_columns,
            var_name="portfolio",
            value_name="weight",
        ).rename(columns={symbol_col: "symbol"})

    wide_df = df.reset_index()
    index_column = wide_df.columns[0]
    wide_df = wide_df.rename(columns={index_column: "symbol"})
    return wide_df.melt(id_vars="symbol", var_name="portfolio", value_name="weight")


def plot_portfolio_weights(
    weights_df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot portfolio weights from tidy or wide allocation data.

    Accepted inputs include tidy frames with ``symbol`` and ``weight`` columns,
    tidy frames with an additional ``portfolio``/``method`` column, or wide
    frames where rows are symbols and columns are portfolio methods.
    """
    plot_df = _coerce_weights_to_long(weights_df)
    _validate_columns(plot_df, {"symbol", "portfolio", "weight"})
    plot_df["weight"] = pd.to_numeric(plot_df["weight"], errors="coerce")
    plot_df = plot_df.dropna(subset=["weight"])
    if plot_df.empty:
        raise ValueError("weights_df contains no numeric weights to plot")

    order = (
        plot_df.groupby("symbol", sort=False)["weight"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    n_symbols = plot_df["symbol"].nunique()
    n_portfolios = plot_df["portfolio"].nunique()
    fig_width = max(9, min(16, n_symbols * max(0.7, 0.3 * n_portfolios) + 4))

    fig, ax = plt.subplots(figsize=(fig_width, 6))
    barplot_kwargs: dict[str, Any] = {
        "data": plot_df,
        "x": "symbol",
        "y": "weight",
        "order": order,
        "ax": ax,
    }
    if n_portfolios > 1:
        barplot_kwargs.update({"hue": "portfolio", "palette": DEFAULT_PALETTE})
    else:
        barplot_kwargs.update({"color": sns.color_palette(DEFAULT_PALETTE)[0]})
    sns.barplot(**barplot_kwargs)
    _apply_common_style(ax, "Portfolio Weights", "Symbol", "Weight")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.tick_params(axis="x", rotation=0)
    if n_portfolios > 1:
        ax.legend(title="Portfolio", frameon=False, ncol=2)
    elif ax.get_legend() is not None:
        ax.get_legend().remove()
    fig.tight_layout()
    _maybe_save_figure(fig, output_path)
    return fig, ax
