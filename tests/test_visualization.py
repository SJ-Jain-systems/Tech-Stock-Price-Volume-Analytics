from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization import (
    plot_adjusted_prices,
    plot_correlation_heatmap,
    plot_cumulative_returns,
    plot_drawdowns,
    plot_portfolio_weights,
    plot_posterior_intervals,
    plot_return_distribution,
    plot_rolling_volatility,
)


def _sample_price_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    frames = []
    for index, symbol in enumerate(["AAPL", "MSFT"]):
        rng = np.random.default_rng(index)
        returns = rng.normal(0.001, 0.015, len(dates))
        prices = 100 * np.cumprod(1 + returns)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "adj_close": prices,
                    "simple_return": pd.Series(prices).pct_change().to_numpy(),
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    result["rolling_ann_vol_63d"] = result.groupby("symbol")["simple_return"].rolling(
        63
    ).std().reset_index(level=0, drop=True) * np.sqrt(252)
    return result


def test_time_series_plot_helpers_return_figures_and_axes() -> None:
    df = _sample_price_frame()

    for plot_func in (
        plot_adjusted_prices,
        plot_cumulative_returns,
        plot_rolling_volatility,
        plot_drawdowns,
        plot_return_distribution,
    ):
        fig, ax = plot_func(df)
        assert fig is ax.figure
        assert ax.get_title()
        assert ax.get_xlabel()
        assert ax.get_ylabel()
        plt.close(fig)


def test_matrix_interval_and_weight_plots_return_figures_and_axes(tmp_path) -> None:
    df = _sample_price_frame()
    return_matrix = df.pivot(index="date", columns="symbol", values="simple_return")
    summary = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "mean": [0.05, 0.08],
            "low": [0.01, 0.03],
            "high": [0.09, 0.12],
        }
    )
    weights = pd.DataFrame(
        {"equal_weight": [0.5, 0.5], "optimized": [0.35, 0.65]},
        index=["AAPL", "MSFT"],
    )

    plot_cases = [
        (plot_correlation_heatmap, (return_matrix.corr(),)),
        (
            plot_posterior_intervals,
            (summary, "mean", "low", "high", "symbol", "Posterior Expected Return"),
        ),
        (plot_portfolio_weights, (weights,)),
    ]

    for index, (plot_func, args) in enumerate(plot_cases):
        output_path = tmp_path / f"plot_{index}.png"
        fig, ax = plot_func(*args, output_path=output_path)
        assert output_path.exists()
        assert fig is ax.figure
        assert ax.get_title()
        plt.close(fig)
