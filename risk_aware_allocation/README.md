# Risk-Aware Portfolio Allocation System

A decision-focused analytics project that forecasts next-period market risk and maps predictions into portfolio allocation changes.

## Status

Phases 1-9 include typed config, data ingestion/validation, feature and label engineering, walk-forward modeling, strategy mapping, and a transaction-cost-aware backtest engine.

## Configuration

Use `get_default_config()` from `src/config.py` to construct a validated V1 configuration object in one place before running the pipeline.

## Sensitivity Analysis

The threshold sensitivity extension evaluates how performance changes across a grid of volatility thresholds while keeping the rest of the pipeline fixed.

Why this matters:
- It tests robustness of the decision rule rather than claiming a single “optimal” threshold.
- It helps identify stable regions where Sharpe, drawdown, and turnover trade-offs are acceptable.
- It reduces the risk of over-interpreting one parameter point as economic truth.

Implemented in `src/sensitivity.py`, this layer:
1. Reuses existing predictions, strategy mapping, backtest, and metrics pipeline.
2. Runs one backtest per threshold.
3. Returns a table including return, volatility, Sharpe, drawdown, Calmar, turnover, transaction costs, and rebalance counts.

Use this output as robustness evidence, not parameter mining.
