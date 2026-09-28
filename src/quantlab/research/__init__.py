"""Reusable statistical research utilities.

Kept separate from the production pipeline on purpose: these are analysis tools a
researcher calls from a notebook or a script, not steps an Airflow DAG runs.  The
separation keeps the pipeline dependency-light and makes it obvious which code
has to be correct for the data to be right, and which has to be correct for the
conclusions to be right.
"""

from quantlab.research.descriptive import correlation_matrix, describe, summary_table
from quantlab.research.hypothesis import (
    diebold_mariano,
    mean_return_t_test,
    sharpe_statistic,
    stationary_bootstrap_ci,
)
from quantlab.research.multiple_testing import (
    adjust_pvalues,
    analyse_sharpe,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from quantlab.research.regression import ols, univariate_screen
from quantlab.research.stationarity import (
    adf_test,
    autocorrelation,
    classify_stationarity,
    kpss_test,
    ljung_box,
)

__all__ = [
    "adf_test",
    "adjust_pvalues",
    "analyse_sharpe",
    "autocorrelation",
    "classify_stationarity",
    "correlation_matrix",
    "deflated_sharpe_ratio",
    "describe",
    "diebold_mariano",
    "kpss_test",
    "ljung_box",
    "mean_return_t_test",
    "ols",
    "probabilistic_sharpe_ratio",
    "sharpe_statistic",
    "stationary_bootstrap_ci",
    "summary_table",
    "univariate_screen",
]
