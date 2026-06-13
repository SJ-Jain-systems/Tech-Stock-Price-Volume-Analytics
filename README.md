# Bayesian Tech Equity Risk Lab

A reproducible financial data science project for uncertainty-aware risk modeling and portfolio analysis of large-cap technology equities.

> **Educational risk-analysis scope.** This project is a descriptive modeling workflow for historical risk estimation, uncertainty-aware analysis, posterior distributions, and educational portfolio comparison. It does not produce trade signals, promise future outcomes, or provide instructions for real capital allocation.

## 1. Project overview

**Bayesian Tech Equity Risk Lab** analyzes historical OHLCV equity data for a focused universe of major technology stocks. The project builds an end-to-end research workflow that starts with raw market data, validates and loads it into DuckDB, engineers financial features with SQL, and then applies Bayesian models to estimate risk and portfolio quantities with posterior uncertainty.

The workflow emphasizes:

- Clean, reproducible data loading and validation.
- SQL-based feature engineering for returns, volatility, drawdowns, liquidity, and portfolio inputs.
- Bayesian models that represent parameter uncertainty and heavy-tailed financial returns.
- Posterior predictive checks and diagnostics instead of relying only on point estimates.
- Portfolio analysis that treats expected return, volatility, downside risk, and correlation as uncertain quantities.

## 2. Why this project matters

Technology equities are liquid, widely followed, and often highly volatile. Traditional analysis frequently summarizes them with historical averages, sample volatility, or single optimized portfolio weights. Those summaries can be useful, but they can also hide meaningful uncertainty—especially when returns are heavy-tailed, regimes change, correlations move together during stress, or downside events dominate investor experience.

This project reframes the analysis around risk and uncertainty:

- **Adjusted-close returns** are used to account for corporate actions such as splits and dividends when available in the dataset.
- **Rolling volatility and drawdowns** expose time-varying risk rather than assuming risk is constant.
- **Bayesian hierarchical modeling** shares information across stocks while still allowing stock-specific behavior.
- **Student-t likelihoods** better accommodate extreme return observations than a simple Gaussian model.
- **Bayesian downside-risk and regime models** focus directly on adverse outcomes and changing market states.
- **Bayesian historical allocation experiments** propagate posterior uncertainty into portfolio-level summaries.

## 3. Dataset description

The project uses a local CSV dataset named `tech_stocks.csv`. The notebooks prefer `data/raw/tech_stocks.csv` and fall back to the repository-root `tech_stocks.csv` when running from a lightweight clone.

Expected source columns:

| Column | Description |
| --- | --- |
| `Unnamed: 0` | Optional source index column; dropped during loading if present. |
| `symbol` | Equity ticker symbol. |
| `date` | Trading date. |
| `open` | Unadjusted opening price. |
| `high` | Unadjusted daily high. |
| `low` | Unadjusted daily low. |
| `close` | Unadjusted closing price. |
| `close_adjusted` | Adjusted close used for return calculations. |
| `volume` | Daily traded share volume. |
| `split_coefficient` | Split adjustment indicator/coefficient from the data source. |

Expected equity universe:

- `AAPL` — Apple
- `AMZN` — Amazon
- `FB` — Meta/Facebook legacy ticker in the dataset
- `GOOG` — Alphabet
- `MSFT` — Microsoft
- `NFLX` — Netflix
- `NVDA` — NVIDIA

## 4. Research questions

The project is organized around uncertainty-aware financial research questions:

1. How clean, complete, and consistent is the raw OHLCV dataset across symbols and dates?
2. How do adjusted-close returns, rolling volatility, drawdowns, and liquidity differ across the technology equity universe?
3. How noisy are sample average returns, and how much shrinkage is introduced by a Bayesian hierarchical return model?
4. Which stocks show the highest posterior volatility, downside-risk probability, and tail-risk exposure?
5. Are there latent return/volatility regimes that help describe market stress and calmer periods?
6. How do posterior uncertainty and correlation uncertainty affect educational portfolio comparison results?
7. Which conclusions are robust under posterior predictive checks, and which are sensitive to model assumptions?

## 5. Project architecture

The repository is structured as a notebook-driven research lab with reusable Python modules and versioned SQL transformations.

```text
Raw CSV data
    ↓
Python validation and DuckDB loading
    ↓
SQL feature-engineering pipeline
    ↓
Financial EDA and risk metric summaries
    ↓
Bayesian return, downside-risk, regime, and portfolio models
    ↓
Posterior diagnostics, figures, dashboard, and final report assets
```

Core layers:

- **Data layer:** CSV loading, schema validation, positive price/volume checks, and DuckDB persistence.
- **SQL layer:** deterministic feature engineering for adjusted-close returns, rolling windows, drawdowns, downside events, and portfolio panels.
- **Modeling layer:** Bayesian return, volatility/downside-risk, regime, and historical allocation experiment notebooks.
- **Reporting layer:** figures, notebook outputs, and final dashboard/report summaries.

## 6. SQL pipeline

The SQL workflow is designed to keep financial feature engineering transparent and reproducible. SQL scripts are executed in order after the raw data is loaded into DuckDB.

| Step | SQL file | Purpose |
| --- | --- | --- |
| 1 | `sql/01_create_tables.sql` | Defines the raw OHLCV price table schema. |
| 2 | `sql/02_clean_prices.sql` | Filters invalid prices/volumes, standardizes `close_adjusted` as `adj_close`, removes duplicate symbol-date rows, and creates a data quality summary. |
| 3 | `sql/03_daily_returns.sql` | Calculates adjusted-close simple returns, log returns, monthly returns, annual returns, and dollar volume. |
| 4 | `sql/04_rolling_features.sql` | Creates rolling momentum, rolling volatility, moving-average, liquidity, and volume z-score features. |
| 5 | `sql/05_risk_metrics.sql` | Builds drawdown tables plus stock-level and annual risk summaries. |
| 6 | `sql/06_portfolio_inputs.sql` | Produces aligned return panels, portfolio summary inputs, and long-form correlation estimates. |

Key SQL outputs include:

- `clean_prices`
- `daily_returns`
- `monthly_returns`
- `annual_returns`
- `rolling_features`
- `drawdowns`
- `stock_level_risk_summary`
- `annual_stock_metrics`
- `portfolio_returns_wide`
- `portfolio_summary_inputs`
- `return_correlation_matrix_long`

## 7. Finance methodology conventions

The project uses a small set of explicit finance conventions so the SQL outputs, notebooks, and report use the same language.

- **Log returns vs. simple returns:** adjusted-close simple return is `adj_close_t / adj_close_{t-1} - 1`; adjusted-close log return is `ln(adj_close_t / adj_close_{t-1})`. Log returns are used for statistical modeling because they are additive through time and work naturally with likelihood-based models. Simple returns are used when an intuitive percentage change, portfolio compounding step, or downside threshold is easier to communicate.
- **Annualization:** expected return is annualized as `mean daily log return × 252`; volatility is annualized as `daily log-return volatility × sqrt(252)`. The 252 factor approximates the number of U.S. trading days in a year.
- **Drawdown:** the running peak is the highest adjusted close observed so far for a stock. Current drawdown is `adj_close / running_peak - 1`, and maximum drawdown is the most negative drawdown in the sample.
- **Value at Risk (VaR):** historical 5% VaR is the 5th percentile of daily returns. Interpreted in return space, a VaR of `-4%` means that 5% of historical days were at or below a 4% loss.
- **Conditional Value at Risk (CVaR):** historical 5% CVaR is the average return on days at or below the 5% VaR threshold. It estimates the expected loss conditional on being in the worst 5% of days.
- **Sharpe-like ratio:** the descriptive risk-adjusted metric is `(annualized return - risk_free_rate) / annualized volatility`, with `risk_free_rate = 0` for simplicity. It should not be read as a formal Sharpe ratio unless an appropriate risk-free rate is included.
- **Correlation and diversification:** pairwise correlations are computed on aligned return dates. Highly correlated technology stocks can move together during broad market or growth-stock stress, limiting diversification even when a portfolio holds several tickers.
- **Tail risk:** the Bayesian return models prefer Student-t likelihoods over normal likelihoods because equity returns often have heavier tails than a Gaussian distribution, making extreme gains and losses more common than a normal model would imply.

## 8. Bayesian methodology

The Bayesian component is focused on estimating financial risk with full uncertainty distributions rather than single historical point estimates.

### Bayesian hierarchical Student-t return model

The return model uses a hierarchical structure across symbols and a Student-t likelihood for daily returns. This combination is useful because technology stock returns can be noisy, heavy-tailed, and cross-sectionally related. The hierarchical prior allows partial pooling across equities while retaining stock-specific posterior estimates.

Primary outputs:

- Posterior expected daily and annualized returns.
- Posterior volatility estimates.
- Shrinkage-aware comparison across symbols.
- Credible intervals for return and risk quantities.
- Posterior predictive checks for model fit.

### Bayesian downside-risk model

The downside-risk workflow focuses on adverse return events rather than only variance. It estimates uncertainty around downside probabilities and tail-sensitive metrics such as bad-day frequency, loss thresholds, and Value-at-Risk-style summaries.

Primary outputs:

- Posterior probability of losses below selected thresholds.
- Stock-level downside-risk intervals.
- Tail-risk comparison across symbols.
- Posterior predictive checks for downside-event behavior.

### Bayesian regime model

The regime model studies whether the return process is better described by latent market states, such as calmer periods and stress periods. The purpose is not to time markets mechanically, but to estimate how return and volatility characteristics differ across states.

Primary outputs:

- Posterior regime probabilities.
- Regime-specific return and volatility summaries.
- Stress-period diagnostics.
- Visualizations of inferred regime behavior over time.

### Bayesian historical allocation experiment

The portfolio workflow uses posterior draws of return, volatility, and covariance-related quantities to evaluate portfolio trade-offs under uncertainty. Instead of presenting a single allocation as definitive, it studies distributions of portfolio risk/return outcomes and weight sensitivity for educational portfolio comparison.

Primary outputs:

- Posterior portfolio return and volatility distributions.
- Portfolio downside-risk summaries.
- Uncertainty-aware efficient-frontier-style analysis.
- Weight stability and concentration diagnostics.

## 9. Financial metrics

The project computes and analyzes financial quantities that are commonly used in empirical asset-risk research:

- **Adjusted-close simple return:** `(adj_close_t / adj_close_{t-1}) - 1`.
- **Adjusted-close log return:** `ln(adj_close_t / adj_close_{t-1})`.
- **Dollar volume:** `adj_close × volume`.
- **Rolling volatility:** sample return volatility over 21, 63, 126, and 252 trading-day windows, with annualized variants.
- **Momentum:** 21, 63, 126, and 252 trading-day adjusted-close returns.
- **Moving averages:** 20, 50, and 200 trading-day price averages.
- **Drawdown:** current adjusted close relative to the running historical peak.
- **Maximum drawdown:** worst drawdown over a selected period.
- **Bad-day frequency:** proportion of days below selected loss thresholds such as -2% and -5%.
- **Historical 5% VaR:** empirical 5th percentile of daily returns, interpreted as a lower-tail loss threshold.
- **Historical 5% CVaR:** average daily return among observations at or below the 5% VaR threshold.
- **Annualized return and volatility:** daily log-return mean and standard deviation scaled by 252 trading days and `sqrt(252)`, respectively.
- **Sharpe-like ratio:** annualized excess return divided by annualized volatility, with `risk_free_rate = 0` in this project unless explicitly changed.
- **Correlation matrix:** pairwise return correlations for aligned portfolio modeling dates.

## 10. Repository structure

```text
Bayesian-Tech-Equity-Risk-Lab/
├── data/
│   ├── raw/                         # Optional local raw data staging
│   ├── processed/                   # Generated feature datasets
│   └── database/                    # Local DuckDB database files
├── models/
│   └── posterior_samples/           # Saved posterior draws and model artifacts
├── notebooks/
│   ├── 01_data_quality_and_sql_loading.ipynb
│   ├── 02_financial_eda.ipynb
│   ├── 03_sql_feature_engineering.ipynb
│   ├── 04_bayesian_return_model.ipynb
│   ├── 05_bayesian_volatility_and_downside_risk.ipynb
│   ├── 06_bayesian_regime_model.ipynb
│   ├── 07_bayesian_portfolio_optimization.ipynb
│   └── 08_final_summary_dashboard.ipynb
├── reports/
│   ├── figures/                     # Generated plots and dashboard assets
│   └── final_report.qmd             # Quarto-compatible final report
├── sql/
│   ├── 01_create_tables.sql
│   ├── 02_clean_prices.sql
│   ├── 03_daily_returns.sql
│   ├── 04_rolling_features.sql
│   ├── 05_risk_metrics.sql
│   └── 06_portfolio_inputs.sql
├── scripts/
│   ├── 01_build_database.py         # Build DuckDB from data/raw/tech_stocks.csv
│   ├── 02_run_sql_pipeline.py       # Execute ordered SQL feature pipeline
│   └── 03_export_processed_features.py # Export processed feature CSV
├── src/
│   ├── config.py                    # Project paths and constants
│   ├── data_loader.py               # CSV loading and validation helpers
│   ├── sql_utils.py                 # DuckDB and SQL pipeline utilities
│   ├── feature_engineering.py       # Python feature-engineering helpers
│   ├── risk_metrics.py              # Financial risk metric helpers
│   ├── bayesian_models.py           # Bayesian modeling helpers
│   ├── portfolio.py                 # Portfolio analytics helpers
│   └── visualization.py             # Plotting utilities
├── tech_stocks.csv                  # Source OHLCV dataset fallback location
├── requirements.txt
└── README.md
```

Generated files such as DuckDB databases, processed datasets, posterior samples, and figures are intended to remain local unless explicitly published.

## 11. How to run the project

### 1. Clone the repository

```bash
git clone <repository-url>
cd Tech-Stock-Price-Volume-Analytics
```

### 2. Create and activate a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Confirm data placement

Use either of the following locations:

```text
data/raw/tech_stocks.csv
```

or the repository-root fallback:

```text
tech_stocks.csv
```

### 5. Run the reproducible script pipeline

The command-line scripts rebuild the DuckDB database, execute the SQL feature pipeline, and export the processed modeling feature dataset. The first script expects the raw CSV at `data/raw/tech_stocks.csv`; if you are using the repository-root fallback dataset, copy it into place first:

```bash
cp tech_stocks.csv data/raw/tech_stocks.csv
python scripts/01_build_database.py
python scripts/02_run_sql_pipeline.py
python scripts/03_export_processed_features.py
```

Script outputs:

- `scripts/01_build_database.py` creates `data/database/tech_stocks.duckdb`, loads `raw_prices`, runs the cleaning SQL, and prints raw/clean row counts plus the data quality summary.
- `scripts/02_run_sql_pipeline.py` runs all SQL scripts in filename order and prints row counts for `clean_prices`, `daily_returns`, `rolling_features`, `drawdowns`, `stock_level_risk_summary`, `annual_stock_metrics`, and `portfolio_returns_wide`.
- `scripts/03_export_processed_features.py` exports `rolling_features` joined with `drawdowns` to `data/processed/tech_stock_features.csv` and prints the output shape and path.

### 6. Run tests

Run the pytest suite from the repository root:

```bash
pytest
```

### 7. Run notebooks in order

Open JupyterLab:

```bash
jupyter lab
```

Then run the notebooks in this sequence:

1. `notebooks/01_data_quality_and_sql_loading.ipynb`
2. `notebooks/02_financial_eda.ipynb`
3. `notebooks/03_sql_feature_engineering.ipynb`
4. `notebooks/04_bayesian_return_model.ipynb`
5. `notebooks/05_bayesian_volatility_and_downside_risk.ipynb`
6. `notebooks/06_bayesian_regime_model.ipynb`
7. `notebooks/07_bayesian_portfolio_optimization.ipynb`
8. `notebooks/08_final_summary_dashboard.ipynb`

The first notebooks create the local DuckDB database and derived SQL tables/views used by later notebooks.

## 12. Key outputs

Expected outputs from a complete project run include:

- Validated data quality summaries by symbol.
- Clean DuckDB tables and views for adjusted-close return analysis.
- Financial EDA charts for prices, returns, volatility, drawdowns, liquidity, and correlations.
- SQL-generated feature tables for rolling volatility, drawdown, downside-risk, and portfolio inputs.
- Bayesian posterior summaries for expected returns and volatility.
- Bayesian downside-risk estimates and uncertainty intervals.
- Bayesian regime probability visualizations and stress-regime summaries.
- Bayesian historical allocation experiment results with uncertainty-aware risk/return trade-offs.
- Posterior predictive checks and model diagnostics.
- Final summary dashboard and report-ready figures under `reports/`.

## 13. Limitations

This project is intentionally research-oriented and has several limitations:

- The equity universe is limited to seven large-cap technology stocks.
- Historical relationships may not persist in future market conditions.
- Adjusted-close data quality depends on the source dataset.
- Daily OHLCV data does not include intraday risk, order book liquidity, option-implied information, or macroeconomic covariates.
- Bayesian models depend on prior choices, likelihood assumptions, sampling diagnostics, and convergence quality.
- Regime models can be sensitive to initialization, number of regimes, and sample period.
- Historical allocation experiment results should be interpreted as scenario and uncertainty analysis, not actionable investment instructions or real capital allocation guidance.
- Transaction costs, taxes, slippage, short-sale constraints, and investor-specific objectives may not be fully represented.

## 14. Future work

Potential extensions include:

- Add automated tests for data validation, SQL pipeline outputs, and risk metric helpers.
- Expand the equity universe and support configurable ticker lists.
- Incorporate Fama-French factors, rates, market indices, sector ETFs, or macro covariates.
- Add Bayesian covariance modeling and dynamic correlation models.
- Compare Student-t models against Gaussian, skewed, and stochastic-volatility alternatives.
- Improve regime modeling with time-varying transition probabilities.
- Add transaction-cost-aware and constraint-aware historical allocation experiments.
- Publish a static HTML report or interactive dashboard artifact.
- Add continuous integration checks for notebook execution and SQL reproducibility.

## 15. Disclaimer

This project is for educational and research purposes only. It uses historical price and volume data and does not account for fundamentals, macroeconomic conditions, taxes, transaction costs, liquidity constraints, or individual investor objectives. Nothing in this project should be interpreted as financial advice.
