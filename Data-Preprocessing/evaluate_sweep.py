#!/usr/bin/env python3
"""Evaluate the theta/run layout with RAPIDS cuDF/cuGraph metrics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from compute_metrics_cpu import (
    calculate_conductance,
    calculate_f1_nmi,
    calculate_graph_structure_profile,
    calculate_inverse_hnorm,
    calculate_modularity,
    load_json_document,
    load_performance_metrics,
)


CONFIG_COLUMNS = [
    "Parameter ID",
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
        description="Compute graph, clustering, and throughput metrics for a paper sweep."
    )
    parser.add_argument("--graph", type=Path, required=True, help="Graph TSV path.")
    parser.add_argument("--results", type=Path, required=True, help="Sweep output root.")
    parser.add_argument("--output", type=Path, required=True, help="Measurement CSV.")
    parser.add_argument(
        "--configurations",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "TopDown-SBP"
        / "configs"
        / "params.conf",
    )
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--dataset", default="Custom")
    parser.add_argument("--result-batch", default="custom")
    parser.add_argument("--num-vertices", type=int, default=None)
    return parser.parse_args()


def read_configurations(path: Path) -> dict[int, dict[str, object]]:
    configurations = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split("|")
            if len(parts) != len(CONFIG_COLUMNS):
                raise SystemExit(
                    f"{path}:{line_number}: expected {len(CONFIG_COLUMNS)} fields, "
                    f"found {len(parts)}."
                )
            row = dict(zip(CONFIG_COLUMNS, parts))
            parameter_id = int(row["Parameter ID"])
            row["Parameter ID"] = parameter_id
            configurations[parameter_id] = row
    return configurations


def read_partition(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=["vertex", "cluster"])


def predicted_partition(document: dict, first_vertex: int) -> pd.DataFrame:
    results = document.get("Results", [])
    if results and isinstance(results[0], (list, tuple)):
        return pd.DataFrame(results, columns=["vertex", "cluster"])
    return pd.DataFrame(
        {
            "vertex": range(first_vertex, first_vertex + len(results)),
            "cluster": results,
        }
    )


def theta_id(path: Path) -> int:
    match = re.fullmatch(r"theta_(\d+)", path.name)
    if not match:
        raise ValueError(path.name)
    return int(match.group(1))


def main() -> None:
    args = parse_args()
    if not args.graph.is_file():
        raise SystemExit(f"Graph not found: {args.graph}")
    if not args.results.is_dir():
        raise SystemExit(f"Sweep result directory not found: {args.results}")
    configurations = read_configurations(args.configurations)

    edges = pd.read_csv(
        args.graph, sep="\t", header=None, names=["source", "target", "weight"]
    )
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(1.0)
    edge_vertices = set(edges["source"].astype(int)) | set(edges["target"].astype(int))
    first_vertex = min(edge_vertices) if edge_vertices else 0
    num_vertices = args.num_vertices or len(edge_vertices)
    graph_profile = calculate_graph_structure_profile(edges, num_vertices)
    truth = read_partition(args.ground_truth) if args.ground_truth else None

    rows = []
    for theta_dir in sorted(args.results.glob("theta_*"), key=theta_id):
        parameter_id = theta_id(theta_dir)
        if parameter_id not in configurations:
            raise SystemExit(f"No configuration found for theta_{parameter_id}.")
        for run_dir in sorted(theta_dir.glob("run_*")):
            result_path = run_dir / "result.json"
            if not result_path.is_file():
                continue
            document = load_json_document(result_path)
            predicted = predicted_partition(document, first_vertex)
            modularity = calculate_modularity(edges, predicted)
            conductance = calculate_conductance(edges, predicted)
            inverse_hnorm = calculate_inverse_hnorm(edges, predicted)
            if truth is None:
                f1_score, nmi = np.nan, np.nan
            else:
                f1_score, nmi = calculate_f1_nmi(predicted, truth, num_vertices)
            performance = load_performance_metrics(str(run_dir), document)
            runtime = performance["runtime_seconds"]
            total_edge_weight = float(edges["weight"].sum())
            teps = (
                (total_edge_weight / runtime) / 1000.0
                if runtime is not None and np.isfinite(runtime) and runtime > 0
                else np.nan
            )
            row = {
                "Dataset": args.dataset,
                "Result Batch": args.result_batch,
                "Graph Name": args.graph.stem,
                "Algorithm": "Top-down SBP",
                "Parameter": f"theta_{parameter_id}",
                "Run": run_dir.name,
                **configurations[parameter_id],
                **graph_profile,
                "Vertices": num_vertices,
                "Edges": total_edge_weight,
                "Directed Modularity": modularity,
                "Directed Conductance": conductance,
                "Normalized MDL": 1.0 - inverse_hnorm,
                "Inverse H_norm": inverse_hnorm,
                "F1 Score": f1_score,
                "NMI": nmi,
                "Runtime": runtime,
                "TEPS": teps,
            }
            rows.append(row)

    if not rows:
        raise SystemExit(f"No result.json files found under {args.results}.")
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Measurement rows: {len(output)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
