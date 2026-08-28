#!/usr/bin/env python3
"""Rank the 27 paper configurations for one or more new graphs."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from workflow import (
    ALPHA_COLUMN,
    CONFIGURATION_COLUMN,
    CONFIGURATION_ID_COLUMN,
    DATASET_COLUMN,
    GRAPH_COLUMN,
    GRAPH_KEY_COLUMN,
    PREDICTED_RANK_COLUMN,
    PREDICTED_UTILITY_COLUMN,
    assign_ranks,
    normalize_degree_product_sort,
    parse_alpha_values,
)


CONFIGURATION_FILE_COLUMNS = [
    CONFIGURATION_ID_COLUMN,
    "SUBGRAPHS",
    "BATCHES",
    "CACHE_SIZE",
    "DEGREEPRODUCTSORT",
    "SPLITINIT",
    "SPLIT",
    "MH_PERCENT",
    "ALGORITHM",
    "OVERLAP",
    "NONPARAMETRIC",
    "NODELTA",
    "MIX",
    "GREEDY",
    "APPROXIMATE",
    "ASYNC_ITERS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict utility and rank Top-down SBP configurations."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--graph-features", type=Path, required=True)
    parser.add_argument("--configurations", type=Path, required=True)
    parser.add_argument(
        "--alpha",
        default="0.5",
        help="One value or a comma-separated list of values in [0, 1].",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_alpha_argument(value: str) -> list[float]:
    try:
        return parse_alpha_values(
            token.strip() for token in value.split(",") if token.strip()
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid --alpha value: {exc}") from exc


def read_configurations(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split("|")
            if len(parts) != len(CONFIGURATION_FILE_COLUMNS):
                raise SystemExit(
                    f"{path}:{line_number}: expected {len(CONFIGURATION_FILE_COLUMNS)} "
                    f"fields, found {len(parts)}."
                )
            rows.append(dict(zip(CONFIGURATION_FILE_COLUMNS, parts)))
    if not rows:
        raise SystemExit(f"No configurations found in {path}.")

    configurations = pd.DataFrame(rows)
    numeric_columns = [
        CONFIGURATION_ID_COLUMN,
        "SUBGRAPHS",
        "BATCHES",
        "CACHE_SIZE",
        "MH_PERCENT",
        "NONPARAMETRIC",
        "NODELTA",
        "MIX",
        "GREEDY",
        "APPROXIMATE",
        "ASYNC_ITERS",
    ]
    for column in numeric_columns:
        configurations[column] = pd.to_numeric(configurations[column], errors="raise")
    try:
        configurations["DEGREEPRODUCTSORT"] = normalize_degree_product_sort(
            configurations["DEGREEPRODUCTSORT"]
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    configurations[CONFIGURATION_ID_COLUMN] = configurations[
        CONFIGURATION_ID_COLUMN
    ].astype(int)
    configurations[CONFIGURATION_COLUMN] = configurations[
        CONFIGURATION_ID_COLUMN
    ].map(lambda value: f"theta_{value}")
    return configurations


def cross_join(
    graphs: pd.DataFrame,
    configurations: pd.DataFrame,
    alphas: list[float],
) -> pd.DataFrame:
    frames = []
    for alpha in alphas:
        candidates = graphs.merge(configurations, how="cross")
        candidates[ALPHA_COLUMN] = alpha
        frames.append(candidates)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    alphas = parse_alpha_argument(args.alpha)
    bundle = joblib.load(args.model)
    required_bundle_keys = {"model", "feature_columns", "graph_property_columns"}
    missing_bundle_keys = required_bundle_keys.difference(bundle)
    if missing_bundle_keys:
        raise SystemExit(f"Model bundle is missing keys: {sorted(missing_bundle_keys)}")

    graphs = pd.read_csv(args.graph_features)
    if GRAPH_COLUMN not in graphs.columns:
        raise SystemExit(f"Graph feature CSV must include {GRAPH_COLUMN!r}.")
    if DATASET_COLUMN not in graphs.columns:
        graphs.insert(0, DATASET_COLUMN, "inference")
    missing_graph_properties = [
        column
        for column in bundle["graph_property_columns"]
        if column not in graphs.columns
    ]
    if missing_graph_properties:
        raise SystemExit(
            f"Graph feature CSV is missing model inputs: {missing_graph_properties}"
        )
    for column in bundle["graph_property_columns"]:
        graphs[column] = pd.to_numeric(graphs[column], errors="raise")
    graphs[GRAPH_KEY_COLUMN] = (
        graphs[DATASET_COLUMN].astype(str) + "::" + graphs[GRAPH_COLUMN].astype(str)
    )

    candidates = cross_join(graphs, read_configurations(args.configurations), alphas)
    missing_features = [
        column for column in bundle["feature_columns"] if column not in candidates.columns
    ]
    if missing_features:
        raise SystemExit(f"Candidate matrix is missing model inputs: {missing_features}")

    candidates[PREDICTED_UTILITY_COLUMN] = bundle["model"].predict(
        candidates[bundle["feature_columns"]]
    )
    candidates = assign_ranks(candidates, PREDICTED_UTILITY_COLUMN)

    output_columns = [
        DATASET_COLUMN,
        GRAPH_COLUMN,
        ALPHA_COLUMN,
        CONFIGURATION_COLUMN,
        CONFIGURATION_ID_COLUMN,
        PREDICTED_UTILITY_COLUMN,
        PREDICTED_RANK_COLUMN,
        *[
            column
            for column in CONFIGURATION_FILE_COLUMNS
            if column != CONFIGURATION_ID_COLUMN
        ],
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates[output_columns].to_csv(args.output, index=False)
    print(f"Recommendations: {args.output}")

    winners = candidates[candidates[PREDICTED_RANK_COLUMN] == 1]
    for _, row in winners.iterrows():
        print(
            f"{row[GRAPH_COLUMN]} alpha={row[ALPHA_COLUMN]:g}: "
            f"{row[CONFIGURATION_COLUMN]} "
            f"(predicted utility={row[PREDICTED_UTILITY_COLUMN]:.6f})"
        )


if __name__ == "__main__":
    main()
