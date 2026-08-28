#!/usr/bin/env python3
"""Evaluate AutoTuneTD-SBP on held-out or external graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from workflow import (
    ALPHA_COLUMN,
    CONFIGURATION_ID_COLUMN,
    GRAPH_KEY_COLUMN,
    PREDICTED_RANK_COLUMN,
    PREDICTED_UTILITY_COLUMN,
    UTILITY_COLUMN,
    assign_ranks,
    build_utility_dataset,
    evaluate_rankings,
    evaluate_rankings_by_alpha,
    parse_alpha_values,
    prepare_averaged_measurements,
    regression_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate predicted configuration rankings using repeated Top-down "
            "SBP measurements."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Measurement CSV.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--partition",
        choices=("test", "validation", "all"),
        default="test",
        help="Use a stored synthetic split or evaluate every graph in the input.",
    )
    parser.add_argument(
        "--alphas",
        default="0.5",
        help="Comma-separated alpha values; the paper's primary result uses 0.5.",
    )
    parser.add_argument("--expected-runs", type=int)
    parser.add_argument("--expected-configurations", type=int)
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--random-state", type=int)
    return parser.parse_args()


def required_bundle(bundle: dict) -> None:
    required = {
        "model",
        "feature_columns",
        "graph_property_columns",
        "alpha_values",
        "accuracy_weights",
        "split_manifest",
        "training_priors",
    }
    missing = required.difference(bundle)
    if missing:
        raise SystemExit(f"Model bundle is missing keys: {sorted(missing)}")


def parse_alpha_argument(value: str) -> list[float]:
    try:
        return parse_alpha_values(
            token.strip() for token in value.split(",") if token.strip()
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid --alphas value: {exc}") from exc


def select_partition(
    averaged: pd.DataFrame, bundle: dict, partition: str
) -> pd.DataFrame:
    if partition == "all":
        return averaged

    expected_keys = set(bundle["split_manifest"][partition])
    available_keys = set(averaged[GRAPH_KEY_COLUMN].astype(str))
    missing = sorted(expected_keys.difference(available_keys))
    if missing:
        raise SystemExit(
            f"Input is missing {len(missing)} graph(s) from the stored {partition} "
            f"partition; examples: {missing[:5]}"
        )
    return averaged[averaged[GRAPH_KEY_COLUMN].isin(expected_keys)].copy()


def prior_ranked_dataset(
    dataset: pd.DataFrame, ordered_ids: list[int]
) -> pd.DataFrame:
    rank_lookup = {int(config_id): rank + 1 for rank, config_id in enumerate(ordered_ids)}
    ranked = dataset.copy()
    ranked[PREDICTED_RANK_COLUMN] = ranked[CONFIGURATION_ID_COLUMN].map(rank_lookup)
    if ranked[PREDICTED_RANK_COLUMN].isna().any():
        missing = sorted(
            ranked.loc[
                ranked[PREDICTED_RANK_COLUMN].isna(), CONFIGURATION_ID_COLUMN
            ].unique()
        )
        raise SystemExit(f"Training prior has no rank for configurations: {missing}")
    ranked[PREDICTED_RANK_COLUMN] = ranked[PREDICTED_RANK_COLUMN].astype(int)
    return ranked


def random_baseline_metrics(
    dataset: pd.DataFrame, repeats: int, random_state: int
) -> tuple[dict[str, float], pd.DataFrame]:
    if repeats <= 0:
        raise SystemExit("--random-repeats must be positive.")
    generator = np.random.default_rng(random_state)
    results = []
    results_by_alpha = []
    for repeat in range(repeats):
        ranked = dataset.copy()
        ranked["Random Score"] = generator.random(len(ranked))
        ranked = assign_ranks(ranked, "Random Score")
        metrics, _ = evaluate_rankings(ranked, method="Random-rank Order")
        results.append(metrics)
        per_alpha = evaluate_rankings_by_alpha(ranked)
        per_alpha["Repeat"] = repeat
        results_by_alpha.append(per_alpha)

    frame = pd.DataFrame(results)
    metrics = {
        column: float(frame[column].mean())
        for column in frame.columns
        if column != "graph_alpha_pairs"
    }
    metrics["graph_alpha_pairs"] = int(results[0]["graph_alpha_pairs"])
    metrics["repeats"] = repeats
    combined = pd.concat(results_by_alpha, ignore_index=True)
    by_alpha = combined.drop(columns="Repeat").groupby(ALPHA_COLUMN, as_index=False).mean()
    by_alpha["graph_alpha_pairs"] = by_alpha["graph_alpha_pairs"].astype(int)
    by_alpha["repeats"] = repeats
    return metrics, by_alpha


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.model)
    required_bundle(bundle)
    alphas = parse_alpha_argument(args.alphas)
    expected_runs = (
        bundle.get("expected_runs", 3)
        if args.expected_runs is None
        else args.expected_runs
    )
    expected_configurations = (
        bundle.get("expected_configurations", 27)
        if args.expected_configurations is None
        else args.expected_configurations
    )
    random_state = (
        bundle.get("random_state", 42)
        if args.random_state is None
        else args.random_state
    )

    try:
        averaged = prepare_averaged_measurements(
            pd.read_csv(args.input),
            bundle["graph_property_columns"],
            expected_runs=expected_runs,
            expected_configurations=expected_configurations,
        )
        averaged = select_partition(averaged, bundle, args.partition)
        dataset = build_utility_dataset(
            averaged, alphas, bundle["accuracy_weights"]
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    missing_features = [
        column for column in bundle["feature_columns"] if column not in dataset.columns
    ]
    if missing_features:
        raise SystemExit(f"Evaluation data is missing model inputs: {missing_features}")

    predictions = dataset.copy()
    predictions[PREDICTED_UTILITY_COLUMN] = bundle["model"].predict(
        predictions[bundle["feature_columns"]]
    )
    predictions = assign_ranks(predictions, PREDICTED_UTILITY_COLUMN)
    autotune_metrics, autotune_selections = evaluate_rankings(predictions)

    method_metrics = {"AutoTuneTD-SBP": autotune_metrics}
    selection_frames = [autotune_selections]
    autotune_by_alpha = evaluate_rankings_by_alpha(predictions)
    autotune_by_alpha.insert(0, "Method", "AutoTuneTD-SBP")
    alpha_metric_frames = [autotune_by_alpha]
    for prior_name, method_name in (
        ("accuracy", "Accuracy-rank Prior"),
        ("performance", "Performance-rank Prior"),
    ):
        prior_ranked = prior_ranked_dataset(
            dataset, bundle["training_priors"][prior_name]
        )
        metrics, selections = evaluate_rankings(prior_ranked, method=method_name)
        method_metrics[method_name] = metrics
        selection_frames.append(selections)
        prior_by_alpha = evaluate_rankings_by_alpha(prior_ranked)
        prior_by_alpha.insert(0, "Method", method_name)
        alpha_metric_frames.append(prior_by_alpha)

    random_metrics, random_by_alpha = random_baseline_metrics(
        dataset, args.random_repeats, random_state
    )
    method_metrics["Random-rank Order"] = random_metrics
    random_by_alpha.insert(0, "Method", "Random-rank Order")
    alpha_metric_frames.append(random_by_alpha)
    method_metrics_by_alpha = pd.concat(alpha_metric_frames, ignore_index=True)
    results = {
        "partition": args.partition,
        "graphs": int(averaged[GRAPH_KEY_COLUMN].nunique()),
        "alpha_values": alphas,
        "utility_regression": regression_metrics(
            predictions[UTILITY_COLUMN], predictions[PREDICTED_UTILITY_COLUMN]
        ),
        "methods": method_metrics,
        "methods_by_alpha": method_metrics_by_alpha.to_dict(orient="records"),
        "random_state": random_state,
    }

    metric_rows = []
    for method, metrics in method_metrics.items():
        metric_rows.append({"Method": method, **metrics})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    pd.concat(selection_frames, ignore_index=True).to_csv(
        args.output_dir / "test_selections.csv", index=False
    )
    pd.DataFrame(metric_rows).to_csv(
        args.output_dir / "test_method_metrics_overall.csv", index=False
    )
    method_metrics_by_alpha.to_csv(
        args.output_dir / "test_method_metrics.csv", index=False
    )
    write_json(args.output_dir / "test_metrics.json", results)

    print(f"Predictions: {args.output_dir / 'test_predictions.csv'}")
    print(f"Metrics: {args.output_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
