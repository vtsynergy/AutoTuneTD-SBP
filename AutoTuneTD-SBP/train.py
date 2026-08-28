#!/usr/bin/env python3
"""Train the AutoTuneTD-SBP utility regressor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from workflow import (
    ALPHA_COLUMN,
    CATEGORICAL_CONFIGURATION_COLUMNS,
    CONFIGURATION_FEATURE_COLUMNS,
    GRAPH_KEY_COLUMN,
    NUMERIC_CONFIGURATION_COLUMNS,
    PREDICTED_UTILITY_COLUMN,
    UTILITY_COLUMN,
    assign_ranks,
    build_training_priors,
    build_utility_dataset,
    evaluate_rankings,
    evaluate_rankings_by_alpha,
    load_graph_property_columns,
    parse_alpha_values,
    prepare_averaged_measurements,
    regression_metrics,
)


DEFAULT_ALPHAS = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average repeated Top-down SBP measurements, compute utility, and "
            "train the XGBoost configuration-ranking model."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Measurement CSV.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--graph-property-manifest",
        type=Path,
        help="Optional text file containing one graph-property column name per line.",
    )
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--w-h", type=float, default=1.0 / 3.0)
    parser.add_argument("--w-q", type=float, default=1.0 / 3.0)
    parser.add_argument("--w-phi", type=float, default=1.0 / 3.0)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--expected-runs", type=int, default=3)
    parser.add_argument("--expected-configurations", type=int, default=27)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--subsample", type=float, default=0.90)
    parser.add_argument("--colsample-bytree", type=float, default=0.90)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--write-training-dataset",
        action="store_true",
        help="Write the graph/configuration/alpha rows used by the model.",
    )
    return parser.parse_args()


def parse_alpha_argument(value: str) -> list[float]:
    try:
        return parse_alpha_values(
            token.strip() for token in value.split(",") if token.strip()
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid --alphas value: {exc}") from exc


def validate_split_fractions(args: argparse.Namespace) -> None:
    fractions = [args.train_fraction, args.validation_fraction, args.test_fraction]
    if any(value <= 0.0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
        raise SystemExit(
            "Train, validation, and test fractions must be positive and sum to 1."
        )


def split_graph_keys(
    graph_keys: np.ndarray,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the paper's graph-level train/validation/test partitions."""
    if len(graph_keys) < 10:
        raise SystemExit(
            "At least 10 graphs are required for a nonempty 60/10/30 graph split."
        )
    training, remainder = train_test_split(
        graph_keys,
        train_size=train_fraction,
        random_state=random_state,
        shuffle=True,
    )
    validation_share = validation_fraction / (validation_fraction + test_fraction)
    validation, test = train_test_split(
        remainder,
        train_size=validation_share,
        random_state=random_state,
        shuffle=True,
    )
    return training, validation, test


def make_model(
    args: argparse.Namespace, graph_property_columns: list[str]
) -> Pipeline:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise SystemExit("Install xgboost before training AutoTuneTD-SBP.") from exc

    numeric_features = [
        *graph_property_columns,
        *NUMERIC_CONFIGURATION_COLUMNS,
        ALPHA_COLUMN,
    ]
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_CONFIGURATION_COLUMNS,
            ),
        ]
    )
    regressor = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        tree_method="hist",
    )
    return Pipeline([("preprocess", preprocessing), ("regressor", regressor)])


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    validate_split_fractions(args)
    alphas = parse_alpha_argument(args.alphas)
    accuracy_weights = (args.w_h, args.w_q, args.w_phi)
    graph_property_columns = load_graph_property_columns(args.graph_property_manifest)

    try:
        averaged = prepare_averaged_measurements(
            pd.read_csv(args.input),
            graph_property_columns,
            expected_runs=args.expected_runs,
            expected_configurations=args.expected_configurations,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    graph_keys = np.array(sorted(averaged[GRAPH_KEY_COLUMN].unique()))
    training_keys, validation_keys, test_keys = split_graph_keys(
        graph_keys,
        args.train_fraction,
        args.validation_fraction,
        args.test_fraction,
        args.random_state,
    )
    training_averages = averaged[averaged[GRAPH_KEY_COLUMN].isin(training_keys)]
    validation_averages = averaged[averaged[GRAPH_KEY_COLUMN].isin(validation_keys)]
    try:
        training_dataset = build_utility_dataset(
            training_averages, alphas, accuracy_weights
        )
        validation = build_utility_dataset(
            validation_averages, alphas, accuracy_weights
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    feature_columns = [
        *graph_property_columns,
        *CONFIGURATION_FEATURE_COLUMNS,
        ALPHA_COLUMN,
    ]

    model = make_model(args, graph_property_columns)
    model.fit(
        training_dataset[feature_columns],
        training_dataset[UTILITY_COLUMN],
    )

    validation[PREDICTED_UTILITY_COLUMN] = model.predict(
        validation[feature_columns]
    )
    validation = assign_ranks(validation, PREDICTED_UTILITY_COLUMN)
    rank_metrics, selections = evaluate_rankings(validation)
    metrics = {
        **regression_metrics(
            validation[UTILITY_COLUMN], validation[PREDICTED_UTILITY_COLUMN]
        ),
        **rank_metrics,
        "graphs_total": int(len(graph_keys)),
        "graphs_training": int(len(training_keys)),
        "graphs_validation": int(len(validation_keys)),
        "graphs_test": int(len(test_keys)),
        "averaged_graph_configuration_rows": int(len(averaged)),
        "training_candidate_rows": int(len(training_dataset)),
        "validation_candidate_rows": int(len(validation)),
        "alpha_values": alphas,
        "random_state": args.random_state,
    }

    split_manifest = {
        "random_state": args.random_state,
        "fractions": {
            "training": args.train_fraction,
            "validation": args.validation_fraction,
            "test": args.test_fraction,
        },
        "training": sorted(str(value) for value in training_keys),
        "validation": sorted(str(value) for value in validation_keys),
        "test": sorted(str(value) for value in test_keys),
    }
    bundle = {
        "format_version": 1,
        "model": model,
        "feature_columns": feature_columns,
        "graph_property_columns": graph_property_columns,
        "configuration_feature_columns": CONFIGURATION_FEATURE_COLUMNS,
        "numeric_configuration_columns": NUMERIC_CONFIGURATION_COLUMNS,
        "categorical_configuration_columns": CATEGORICAL_CONFIGURATION_COLUMNS,
        "alpha_values": alphas,
        "accuracy_weights": accuracy_weights,
        "expected_runs": args.expected_runs,
        "expected_configurations": args.expected_configurations,
        "split_manifest": split_manifest,
        "training_priors": build_training_priors(
            training_averages, accuracy_weights
        ),
        "random_state": args.random_state,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_dir / "model.joblib")
    validation.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    selections.to_csv(args.output_dir / "validation_selections.csv", index=False)
    evaluate_rankings_by_alpha(validation).to_csv(
        args.output_dir / "validation_metrics_by_alpha.csv", index=False
    )
    write_json(args.output_dir / "validation_metrics.json", metrics)
    write_json(args.output_dir / "split_manifest.json", split_manifest)
    if args.write_training_dataset:
        training_dataset.to_csv(args.output_dir / "training_dataset.csv", index=False)

    print(f"Model: {args.output_dir / 'model.joblib'}")
    print(f"Validation metrics: {args.output_dir / 'validation_metrics.json'}")
    print(f"Split manifest: {args.output_dir / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
