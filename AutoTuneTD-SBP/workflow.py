"""Shared data preparation and ranking utilities for AutoTuneTD-SBP."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, r2_score


DATASET_COLUMN = "Dataset"
GRAPH_COLUMN = "Graph Name"
GRAPH_KEY_COLUMN = "Graph Key"
CONFIGURATION_COLUMN = "Configuration"
CONFIGURATION_ID_COLUMN = "Configuration ID"
ALPHA_COLUMN = "Alpha"

ACCURACY_COLUMN = "Clustering Accuracy"
PERFORMANCE_COLUMN = "Computational Performance"
NORMALIZED_ACCURACY_COLUMN = "Normalized Accuracy"
NORMALIZED_PERFORMANCE_COLUMN = "Normalized Performance"
UTILITY_COLUMN = "Utility Score"
PREDICTED_UTILITY_COLUMN = "Predicted Utility Score"
PREDICTED_RANK_COLUMN = "Predicted Rank"

OUTCOME_COLUMNS = ["Inverse H_norm", "Directed Modularity", "Directed Conductance", "TEPS"]

# These are the graph properties emitted by the repository's preprocessing pipeline.
GRAPH_PROPERTY_COLUMNS = [
    "Vertices",
    "Edges",
    "Max In-Degree",
    "Avg In-Degree",
    "Max Out-Degree",
    "Avg Out-Degree",
    "Avg CC",
    "Max Assortativity (knn)",
    "Avg Assortativity (knn)",
]
for percentile in range(10, 100, 10):
    GRAPH_PROPERTY_COLUMNS.extend(
        [
            f"In-Degree {percentile}th",
            f"Out-Degree {percentile}th",
            f"CC {percentile}th",
            f"Assortativity {percentile}th",
        ]
    )

# Only parameters varied by the paper are model inputs. Fixed execution settings are
# retained in configuration files but do not provide information to the regressor.
NUMERIC_CONFIGURATION_COLUMNS = [
    "BATCHES",
    "CACHE_SIZE",
    "MH_PERCENT",
]
CATEGORICAL_CONFIGURATION_COLUMNS = [
    "DEGREEPRODUCTSORT",
    "SPLITINIT",
    "SPLIT",
    "NODELTA",
    "NONPARAMETRIC",
    "GREEDY",
    "APPROXIMATE",
    "MIX",
]
CONFIGURATION_FEATURE_COLUMNS = (
    NUMERIC_CONFIGURATION_COLUMNS + CATEGORICAL_CONFIGURATION_COLUMNS
)
PAPER_CONFIGURATION_IDS = tuple(range(1, 28))


def parse_alpha_values(values: Iterable[float]) -> list[float]:
    """Return unique alpha values after checking the paper's [0, 1] domain."""
    alphas = sorted({float(value) for value in values})
    if not alphas or any(value < 0.0 or value > 1.0 for value in alphas):
        raise ValueError("Alpha values must be in the closed interval [0, 1].")
    return alphas


def load_graph_property_columns(path: Path | None) -> list[str]:
    """Load an optional one-column feature manifest."""
    if path is None:
        return list(GRAPH_PROPERTY_COLUMNS)

    columns = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not columns:
        raise ValueError(f"Graph-property manifest is empty: {path}")
    if len(columns) != len(set(columns)):
        raise ValueError(f"Graph-property manifest contains duplicate names: {path}")
    return columns


def _configuration_id(value: object) -> int:
    match = re.search(r"(\d+)\s*$", str(value))
    if match is None:
        raise ValueError(f"Cannot derive a configuration ID from {value!r}.")
    return int(match.group(1))


def normalize_degree_product_sort(values: pd.Series) -> pd.Series:
    """Represent the degree-product-sort switch consistently across file formats."""
    aliases = {
        "0": "off",
        "0.0": "off",
        "false": "off",
        "off": "off",
        "1": "on",
        "1.0": "on",
        "true": "on",
        "on": "on",
    }
    normalized = values.astype(str).str.strip().str.lower().map(aliases)
    if normalized.isna().any():
        invalid = sorted(values[normalized.isna()].astype(str).unique())
        raise ValueError(f"Invalid DEGREEPRODUCTSORT values: {invalid}")
    return normalized


def normalize_measurement_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map historical CSV column names to the terminology used in the paper."""
    data = frame.copy()
    aliases = {
        "Result Batch": DATASET_COLUMN,
        "Parameter": CONFIGURATION_COLUMN,
        "Parameter ID": CONFIGURATION_ID_COLUMN,
        "SUBGRAPHPARTITION": "DEGREEPRODUCTSORT",
    }
    for old_name, new_name in aliases.items():
        if new_name not in data.columns and old_name in data.columns:
            data = data.rename(columns={old_name: new_name})

    if DATASET_COLUMN not in data.columns:
        data.insert(0, DATASET_COLUMN, "measurements")

    if CONFIGURATION_ID_COLUMN not in data.columns:
        if CONFIGURATION_COLUMN not in data.columns:
            raise ValueError(
                "Input must contain Configuration ID or a Configuration/Parameter label."
            )
        data[CONFIGURATION_ID_COLUMN] = data[CONFIGURATION_COLUMN].map(_configuration_id)

    data[CONFIGURATION_ID_COLUMN] = pd.to_numeric(
        data[CONFIGURATION_ID_COLUMN], errors="raise"
    ).astype(int)
    if "DEGREEPRODUCTSORT" in data.columns:
        data["DEGREEPRODUCTSORT"] = normalize_degree_product_sort(
            data["DEGREEPRODUCTSORT"]
        )
    data[CONFIGURATION_COLUMN] = data[CONFIGURATION_ID_COLUMN].map(
        lambda value: f"theta_{value}"
    )
    return data


def prepare_averaged_measurements(
    frame: pd.DataFrame,
    graph_property_columns: Sequence[str],
    expected_runs: int = 3,
    expected_configurations: int = 27,
) -> pd.DataFrame:
    """Validate measurements and average repeated runs by graph/configuration."""
    data = normalize_measurement_columns(frame)
    data = data[data[CONFIGURATION_ID_COLUMN].isin(PAPER_CONFIGURATION_IDS)].copy()
    if data.empty:
        raise ValueError("Input does not contain any of the 27 paper configurations.")
    required_columns = [
        DATASET_COLUMN,
        GRAPH_COLUMN,
        CONFIGURATION_ID_COLUMN,
        *graph_property_columns,
        *CONFIGURATION_FEATURE_COLUMNS,
        *OUTCOME_COLUMNS,
    ]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")

    numeric_columns = [
        CONFIGURATION_ID_COLUMN,
        *graph_property_columns,
        *NUMERIC_CONFIGURATION_COLUMNS,
        *OUTCOME_COLUMNS,
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if data[[DATASET_COLUMN, GRAPH_COLUMN]].isna().any().any():
        raise ValueError("Dataset and Graph Name cannot contain missing values.")

    data[GRAPH_KEY_COLUMN] = (
        data[DATASET_COLUMN].astype(str) + "::" + data[GRAPH_COLUMN].astype(str)
    )
    group_columns = [GRAPH_KEY_COLUMN, CONFIGURATION_ID_COLUMN]

    invariant_columns = CONFIGURATION_FEATURE_COLUMNS
    inconsistent = []
    for column in invariant_columns:
        counts = data.groupby(group_columns, sort=False)[column].nunique(dropna=False)
        if (counts > 1).any():
            inconsistent.append(column)
    if inconsistent:
        raise ValueError(
            "Configuration values change across repeated runs for: "
            + ", ".join(inconsistent)
        )

    run_counts = data.groupby(group_columns, sort=False).size()
    if expected_runs > 0 and not (run_counts == expected_runs).all():
        examples = run_counts[run_counts != expected_runs].head(5).to_dict()
        raise ValueError(
            f"Expected {expected_runs} runs per graph/configuration; mismatches include "
            f"{examples}. Use --expected-runs 0 only for incomplete exploratory data."
        )

    aggregation = {
        DATASET_COLUMN: "first",
        GRAPH_COLUMN: "first",
        **{column: "first" for column in graph_property_columns},
        **{column: "first" for column in CONFIGURATION_FEATURE_COLUMNS},
        **{column: "mean" for column in OUTCOME_COLUMNS},
    }
    averaged = data.groupby(group_columns, as_index=False, sort=False).agg(aggregation)
    averaged = averaged.merge(
        run_counts.rename("Run Count").reset_index(), on=group_columns, how="left"
    )
    averaged[CONFIGURATION_COLUMN] = averaged[CONFIGURATION_ID_COLUMN].map(
        lambda value: f"theta_{int(value)}"
    )

    configuration_counts = averaged.groupby(GRAPH_KEY_COLUMN)[
        CONFIGURATION_ID_COLUMN
    ].nunique()
    if expected_configurations > 0 and not (
        configuration_counts == expected_configurations
    ).all():
        examples = configuration_counts[
            configuration_counts != expected_configurations
        ].head(5).to_dict()
        raise ValueError(
            f"Expected {expected_configurations} configurations per graph; mismatches "
            f"include {examples}. Use --expected-configurations 0 only for incomplete "
            "exploratory data."
        )
    return averaged


def _validate_accuracy_weights(weights: Sequence[float]) -> tuple[float, float, float]:
    if len(weights) != 3:
        raise ValueError(
            "Accuracy requires three weights: description length, modularity, "
            "conductance."
        )
    weight_tuple = tuple(float(value) for value in weights)
    if any(value < 0.0 for value in weight_tuple) or not np.isclose(
        sum(weight_tuple), 1.0
    ):
        raise ValueError("Accuracy weights must be nonnegative and sum to 1.")
    return weight_tuple


def add_accuracy_and_performance(
    averaged: pd.DataFrame, accuracy_weights: Sequence[float]
) -> pd.DataFrame:
    """Compute the accuracy and performance terms defined in the paper."""
    weight_h, weight_q, weight_phi = _validate_accuracy_weights(accuracy_weights)
    data = averaged.copy()
    data[ACCURACY_COLUMN] = (
        weight_h * data["Inverse H_norm"]
        + weight_q * data["Directed Modularity"]
        + weight_phi * (1.0 - data["Directed Conductance"])
    )
    # TEPS is stored as thousands of processed edges per second.
    data[PERFORMANCE_COLUMN] = data["TEPS"]
    return data


def build_utility_dataset(
    averaged: pd.DataFrame,
    alphas: Sequence[float],
    accuracy_weights: Sequence[float],
) -> pd.DataFrame:
    """Compute normalized accuracy, normalized performance, and utility."""
    data = add_accuracy_and_performance(averaged, accuracy_weights)
    accuracy_maximum = data.groupby(GRAPH_KEY_COLUMN)[ACCURACY_COLUMN].transform("max")
    performance_maximum = data.groupby(GRAPH_KEY_COLUMN)[PERFORMANCE_COLUMN].transform(
        "max"
    )
    if (accuracy_maximum <= 0.0).any() or (performance_maximum <= 0.0).any():
        raise ValueError(
            "Accuracy and performance maxima must be positive for normalization."
        )

    data[NORMALIZED_ACCURACY_COLUMN] = (
        np.exp(data[ACCURACY_COLUMN] / accuracy_maximum) - 1.0
    ) / (np.e - 1.0)
    data[NORMALIZED_PERFORMANCE_COLUMN] = data[PERFORMANCE_COLUMN] / performance_maximum

    utility_frames = []
    for alpha in parse_alpha_values(alphas):
        alpha_frame = data.copy()
        alpha_frame[ALPHA_COLUMN] = alpha
        alpha_frame[UTILITY_COLUMN] = (
            alpha * alpha_frame[NORMALIZED_ACCURACY_COLUMN]
            + (1.0 - alpha) * alpha_frame[NORMALIZED_PERFORMANCE_COLUMN]
        )
        utility_frames.append(alpha_frame)
    return pd.concat(utility_frames, ignore_index=True)


def build_training_priors(
    averaged_training_data: pd.DataFrame,
    accuracy_weights: Sequence[float],
) -> dict[str, list[int]]:
    """Build fixed accuracy and performance rankings from training-graph winners."""
    data = add_accuracy_and_performance(averaged_training_data, accuracy_weights)
    all_ids = sorted(data[CONFIGURATION_ID_COLUMN].astype(int).unique())
    priors: dict[str, list[int]] = {}
    for name, value_column in (
        ("accuracy", ACCURACY_COLUMN),
        ("performance", PERFORMANCE_COLUMN),
    ):
        winners = (
            data.sort_values(
                [GRAPH_KEY_COLUMN, value_column, CONFIGURATION_ID_COLUMN],
                ascending=[True, False, True],
            )
            .groupby(GRAPH_KEY_COLUMN, sort=False)
            .first()[CONFIGURATION_ID_COLUMN]
        )
        winner_counts = winners.value_counts().to_dict()
        priors[name] = sorted(
            all_ids, key=lambda config_id: (-winner_counts.get(config_id, 0), config_id)
        )
    return priors


def assign_ranks(
    frame: pd.DataFrame,
    score_column: str,
    rank_column: str = PREDICTED_RANK_COLUMN,
) -> pd.DataFrame:
    """Assign a deterministic rank for each graph and alpha."""
    ranked = frame.sort_values(
        [GRAPH_KEY_COLUMN, ALPHA_COLUMN, score_column, CONFIGURATION_ID_COLUMN],
        ascending=[True, True, False, True],
    ).copy()
    ranked[rank_column] = (
        ranked.groupby([GRAPH_KEY_COLUMN, ALPHA_COLUMN], sort=False).cumcount() + 1
    )
    return ranked


def regression_metrics(observed: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Return utility-regression diagnostics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
    }


def _relative_error(group: pd.DataFrame, column: str, selected_index: object) -> float:
    maximum = float(group[column].max())
    minimum = float(group[column].min())
    if np.isclose(maximum, minimum):
        return 0.0
    selected = float(group.loc[selected_index, column])
    return (maximum - selected) / (maximum - minimum)


def evaluate_rankings(
    ranked: pd.DataFrame,
    rank_column: str = PREDICTED_RANK_COLUMN,
    method: str = "AutoTuneTD-SBP",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate predicted configuration rankings using the paper's metrics."""
    records = []
    selections = []
    group_columns = [GRAPH_KEY_COLUMN, ALPHA_COLUMN]
    for (_, alpha), group in ranked.groupby(group_columns, sort=False):
        observed = group.sort_values(
            [UTILITY_COLUMN, CONFIGURATION_ID_COLUMN], ascending=[False, True]
        )
        predicted = group.sort_values(
            [rank_column, CONFIGURATION_ID_COLUMN], ascending=[True, True]
        )
        observed_ids = observed[CONFIGURATION_ID_COLUMN].astype(int).tolist()
        predicted_ids = predicted[CONFIGURATION_ID_COLUMN].astype(int).tolist()
        observed_ranks = {config_id: rank + 1 for rank, config_id in enumerate(observed_ids)}
        predicted_ranks = {config_id: rank + 1 for rank, config_id in enumerate(predicted_ids)}
        aligned_ids = sorted(observed_ranks)
        tau = kendalltau(
            [observed_ranks[config_id] for config_id in aligned_ids],
            [predicted_ranks[config_id] for config_id in aligned_ids],
        ).statistic

        relevance = group[UTILITY_COLUMN].to_numpy(dtype=float)[None, :]
        ranking_scores = -group[rank_column].to_numpy(dtype=float)[None, :]
        ndcg = ndcg_score(relevance, ranking_scores, k=min(5, len(group)))

        selected_index = predicted.index[0]
        selected_id = int(group.loc[selected_index, CONFIGURATION_ID_COLUMN])
        best_id = observed_ids[0]
        utility_error = _relative_error(group, UTILITY_COLUMN, selected_index)
        accuracy_error = _relative_error(group, ACCURACY_COLUMN, selected_index)
        performance_error = _relative_error(group, PERFORMANCE_COLUMN, selected_index)
        records.append(
            {
                "top_1": float(selected_id == best_id),
                "top_5": float(best_id in predicted_ids[:5]),
                "ndcg_at_5": float(ndcg),
                "kendall_tau": float(tau),
                "relative_utility_error": utility_error,
                "relative_accuracy_error": accuracy_error,
                "relative_performance_error": performance_error,
            }
        )
        selections.append(
            {
                "Method": method,
                DATASET_COLUMN: group.iloc[0][DATASET_COLUMN],
                GRAPH_COLUMN: group.iloc[0][GRAPH_COLUMN],
                ALPHA_COLUMN: float(alpha),
                "Selected Configuration": f"theta_{selected_id}",
                "Observed Best Configuration": f"theta_{best_id}",
                "Relative Utility Error": utility_error,
                "Relative Accuracy Error": accuracy_error,
                "Relative Performance Error": performance_error,
            }
        )

    metric_frame = pd.DataFrame.from_records(records)
    metrics = {column: float(metric_frame[column].mean()) for column in metric_frame}
    metrics["graph_alpha_pairs"] = int(len(metric_frame))
    return metrics, pd.DataFrame.from_records(selections)


def evaluate_rankings_by_alpha(
    ranked: pd.DataFrame,
    rank_column: str = PREDICTED_RANK_COLUMN,
) -> pd.DataFrame:
    """Return one row of ranking metrics for each alpha value."""
    rows = []
    for alpha in sorted(ranked[ALPHA_COLUMN].unique()):
        metrics, _ = evaluate_rankings(
            ranked[ranked[ALPHA_COLUMN] == alpha], rank_column=rank_column
        )
        rows.append({ALPHA_COLUMN: float(alpha), **metrics})
    return pd.DataFrame(rows)
