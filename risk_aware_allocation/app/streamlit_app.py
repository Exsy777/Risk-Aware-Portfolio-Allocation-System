"""Streamlit dashboard for the Risk-Aware Portfolio Allocation project."""

from __future__ import annotations

import copy
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from src.backtest import run_backtest
from src.config import ProjectConfig, get_default_config
from src.data_loader import get_aligned_market_data
from src.features import build_feature_frame
from src.labels import build_label_frame
from src.metrics import summarize_backtest_result
from src.model import run_walk_forward_model
from src.plots import (
    plot_drawdowns,
    plot_equity_curves,
    plot_predictions_vs_actuals,
    plot_threshold_sensitivity_drawdown,
    plot_threshold_sensitivity_sharpe,
    plot_threshold_sensitivity_turnover,
    plot_weights,
)
from src.sensitivity import run_threshold_sensitivity, summarize_threshold_results
from src.strategy import run_strategy_from_predictions


def _build_runtime_config(
    base_config: ProjectConfig,
    start_date: date,
    end_date: date,
    threshold: float,
    transaction_cost_bps: float,
) -> ProjectConfig:
    """Create a runtime config copy based on dashboard controls."""
    runtime_config = copy.deepcopy(base_config)
    runtime_config.data.start_date = start_date.isoformat()
    runtime_config.data.end_date = end_date.isoformat()
    runtime_config.strategy.volatility_threshold = threshold
    runtime_config.strategy.transaction_cost_bps = transaction_cost_bps
    runtime_config.validate()
    return runtime_config


def _run_pipeline(
    config: ProjectConfig,
    model_name: str,
    threshold_grid: list[float],
) -> dict[str, object]:
    """Run the end-to-end pipeline for dashboard display."""
    market_data = get_aligned_market_data(config=config, use_cache=True, save_processed=True)
    features = build_feature_frame(prices=market_data.prices, returns=market_data.returns, config=config)
    labels = build_label_frame(returns=market_data.returns, config=config)

    model_result = run_walk_forward_model(
        features=features,
        labels=labels,
        config=config,
        model_name=model_name,
    )
    strategy_result = run_strategy_from_predictions(model_result.predictions, config=config)
    backtest_result = run_backtest(returns=market_data.returns, strategy_result=strategy_result, config=config)

    metrics_table = summarize_backtest_result(backtest_result)

    sensitivity_results = run_threshold_sensitivity(
        predictions=model_result.predictions,
        returns=market_data.returns,
        config=config,
        threshold_values=threshold_grid,
    )
    sensitivity_summary = summarize_threshold_results(sensitivity_results)

    return {
        "model_result": model_result,
        "backtest_result": backtest_result,
        "metrics_table": metrics_table,
        "sensitivity_results": sensitivity_results,
        "sensitivity_summary": sensitivity_summary,
    }


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title="Risk-Aware Portfolio Allocation Dashboard", layout="wide")

    st.title("Risk-Aware Portfolio Allocation Dashboard")
    st.write(
        "This system forecasts next-period market risk and adjusts portfolio exposure "
        "between SPY and SHY to test whether dynamic de-risking improves risk-adjusted "
        "outcomes relative to static benchmark portfolios."
    )

    st.header("Investor Objective")
    st.write(
        "This project is designed for a risk-aware long-term investor seeking better downside "
        "control than a static benchmark allocation. The focus is decision support: translating "
        "risk forecasts into allocation actions, not raw return prediction alone."
    )

    default_config = get_default_config()

    with st.sidebar:
        st.header("Controls")
        model_name = st.selectbox(
            "Model Choice",
            options=["linear_regression", "random_forest"],
            index=0,
        )

        start_date = st.date_input(
            "Start Date",
            value=pd.to_datetime(default_config.data.start_date).date(),
        )
        end_date = st.date_input("End Date", value=date.today())

        threshold = st.slider(
            "Volatility Threshold",
            min_value=0.05,
            max_value=0.50,
            value=float(default_config.strategy.volatility_threshold),
            step=0.01,
        )

        transaction_cost_bps = st.number_input(
            "Transaction Cost (bps)",
            min_value=0.0,
            max_value=100.0,
            value=float(default_config.strategy.transaction_cost_bps),
            step=1.0,
        )

        st.subheader("Threshold Sensitivity Grid")
        threshold_min = st.number_input("Min Threshold", value=0.10, step=0.01, format="%.2f")
        threshold_max = st.number_input("Max Threshold", value=0.30, step=0.01, format="%.2f")
        threshold_points = st.slider("Number of Grid Points", min_value=3, max_value=21, value=9)

    st.header("Method Overview")
    st.markdown(
        "- **Assets and Data:** Daily SPY and SHY market data from Yahoo Finance.\n"
        "- **Features:** Leakage-safe trailing return, volatility, moving-average, drawdown, and downside-risk features.\n"
        "- **Label:** 5-day forward realized volatility of SPY.\n"
        "- **Validation:** Walk-forward out-of-sample training and prediction only.\n"
        "- **Decision Rule:** Predicted risk above threshold shifts to defensive allocation; below threshold shifts to risk-on allocation.\n"
        "- **Backtest Timing:** Weights decided at date t are applied starting on the next return date (no same-day look-ahead)."
    )

    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        return

    if threshold_min >= threshold_max:
        st.error("Sensitivity min threshold must be less than max threshold.")
        return

    threshold_grid = np.linspace(threshold_min, threshold_max, threshold_points).tolist()

    try:
        runtime_config = _build_runtime_config(
            base_config=default_config,
            start_date=start_date,
            end_date=end_date,
            threshold=threshold,
            transaction_cost_bps=transaction_cost_bps,
        )
        outputs = _run_pipeline(
            config=runtime_config,
            model_name=model_name,
            threshold_grid=threshold_grid,
        )
    except Exception as exc:
        st.error("Unable to run pipeline with current environment/settings.")
        st.exception(exc)
        return

    model_result = outputs["model_result"]
    backtest_result = outputs["backtest_result"]
    metrics_table = outputs["metrics_table"]
    sensitivity_results = outputs["sensitivity_results"]
    sensitivity_summary = outputs["sensitivity_summary"]

    st.header("Main Performance Results")
    st.write(
        "Comparison of dynamic strategy and static benchmarks on annualized return, risk, "
        "drawdown, turnover, and transaction-cost-aware outcomes."
    )
    st.dataframe(metrics_table, use_container_width=True)

    st.header("Main Charts")
    st.pyplot(
        plot_equity_curves(
            strategy_equity=backtest_result.strategy_equity_curve,
            benchmark_equity_curves=backtest_result.benchmark_equity_curves,
        )
    )
    st.pyplot(
        plot_drawdowns(
            strategy_equity=backtest_result.strategy_equity_curve,
            benchmark_equity_curves=backtest_result.benchmark_equity_curves,
        )
    )
    st.pyplot(plot_weights(backtest_result.strategy_daily_weights))
    st.pyplot(
        plot_predictions_vs_actuals(
            predictions=model_result.predictions,
            actuals=model_result.actuals,
        )
    )

    st.header("Threshold Sensitivity (Robustness Analysis)")
    st.write(
        "Threshold sensitivity is presented as robustness analysis. The goal is to inspect "
        "trade-offs across plausible thresholds, not to mine a single best parameter."
    )
    st.dataframe(sensitivity_results, use_container_width=True)
    st.json(sensitivity_summary)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.pyplot(plot_threshold_sensitivity_sharpe(sensitivity_results))
    with col2:
        st.pyplot(plot_threshold_sensitivity_drawdown(sensitivity_results))
    with col3:
        st.pyplot(plot_threshold_sensitivity_turnover(sensitivity_results))

    st.header("Interpretation")
    st.markdown(
        "- Lower thresholds may shift defensive more often.\n"
        "- Higher thresholds may preserve more upside but can react more slowly to rising risk.\n"
        "- Dynamic allocation can improve drawdown control but may trade off upside capture or add turnover.\n"
        "- Sensitivity analysis should be read as robustness analysis, not parameter mining."
    )

    st.header("Limitations")
    st.markdown(
        "- Volatility is difficult to predict consistently.\n"
        "- Results depend on historical relationships that may change.\n"
        "- Threshold choice meaningfully affects behavior and outcomes.\n"
        "- SPY/SHY is a simplified asset universe.\n"
        "- Backtests do not guarantee future outcomes."
    )


if __name__ == "__main__":
    main()
