#!/usr/bin/env python3
"""Generate the 6,000-graph SBM corpus used in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset_manifest import (
    SBM_COMMUNITY_COUNTS,
    SBM_GRAPHS_PER_STRATUM,
    SBM_SCALES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/SBM"),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the twelve generation strata without importing RAPIDS.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing generated graph files to be replaced.",
    )
    return parser.parse_args()


def corpus_plan() -> list[tuple[int, int, int]]:
    return [
        (nodes, edges, communities)
        for nodes, edges in SBM_SCALES
        for communities in SBM_COMMUNITY_COUNTS
    ]


def print_plan(output_dir: Path) -> None:
    print("Paper SBM corpus")
    print(f"Output directory: {output_dir}")
    for index, (nodes, edges, communities) in enumerate(corpus_plan(), start=1):
        print(
            f"{index:02d}: nodes={nodes:,}, edges={edges:,}, "
            f"communities={communities}, graphs={SBM_GRAPHS_PER_STRATUM}"
        )
    print(f"Total graphs: {len(corpus_plan()) * SBM_GRAPHS_PER_STRATUM:,}")


def generation_arguments(
    output_dir: Path,
    nodes: int,
    edges: int,
    communities: int,
    workers: int,
    seed: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=str(output_dir),
        num_nodes=nodes,
        num_edges=edges,
        num_communities=communities,
        num_graphs=SBM_GRAPHS_PER_STRATUM,
        candidate_multiplier=10,
        sampling_strategy="planned",
        degree_mode="mixed",
        target_mu=None,
        workers=workers,
        bins=10,
        seed=seed,
    )


def main() -> None:
    args = parse_args()
    print_plan(args.output_dir)
    if args.dry_run:
        return
    if args.workers < 1:
        raise SystemExit("--workers must be positive.")

    existing_graphs = list(args.output_dir.glob("SBPGraph_N*_M*_K*_id*.tsv"))
    if existing_graphs and not args.overwrite:
        raise SystemExit(
            f"{args.output_dir} already contains generated SBM files; use "
            "--overwrite to replace them."
        )

    try:
        from generate_synthetic_graphs import main as generate_stratum
    except ImportError as exc:
        raise SystemExit(
            "The paper-scale SBM generator requires the RAPIDS environment."
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (nodes, edges, communities) in enumerate(corpus_plan()):
        print(
            f"Generating stratum {index + 1}/{len(corpus_plan())}: "
            f"N={nodes}, M={edges}, K={communities}"
        )
        stratum_args = generation_arguments(
            args.output_dir,
            nodes,
            edges,
            communities,
            args.workers,
            args.seed + index * SBM_GRAPHS_PER_STRATUM,
        )
        generate_stratum(stratum_args)

    expected_count = len(corpus_plan()) * SBM_GRAPHS_PER_STRATUM
    generated_count = len(
        [
            path
            for path in args.output_dir.glob("SBPGraph_N*_M*_K*_id*.tsv")
            if not path.stem.endswith("_truePartition")
        ]
    )
    if generated_count != expected_count:
        raise SystemExit(
            f"Expected {expected_count:,} SBM graphs, found {generated_count:,}."
        )
    print(f"Generated SBM graphs: {generated_count:,}")


if __name__ == "__main__":
    main()
