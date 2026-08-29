#!/usr/bin/env python3
"""Paper dataset manifest and local-layout validator."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path


# External data sources:
#
# MIT Graph Challenge:
#   Download the 2022 static stochastic-block-partition graphs from
#   https://graphchallenge.mit.edu/data-sets/ or the public
#   s3://graphchallenge bucket. Use the adjacency TSV files, not MMIO or
#   incidence matrices. Graph Challenge TSV matrices use 1-based vertex IDs.
#
# SNAP:
#   Download the five source edge lists from https://snap.stanford.edu/data/.
#   The required files are com-dblp.ungraph.txt.gz,
#   com-youtube.ungraph.txt.gz, wiki-topcats.txt.gz,
#   com-lj.ungraph.txt.gz, and com-orkut.ungraph.txt.gz. SNAP files contain
#   two whitespace-separated vertex IDs and comment lines beginning with '#'.
#   convert_edge_list.py adds the required unit weight.
#
# CAIDA:
#   Request authorized UCSD Network Telescope access through
#   https://www.caida.org/catalog/datasets/telescope-near-real-time_dataset/.
#   Raw PCAP is not accepted by this repository. Construct one anonymized,
#   weighted directed edge list per daily graph under the applicable CAIDA
#   agreement. Do not commit raw packets, IP addresses, or sensitive derivatives.
#
# Canonical graph format used by Top-down SBP and the metric scripts:
#   source<TAB>destination<TAB>weight
# The file has no header. Vertex IDs are integers and weight is numeric. A known
# non-overlapping partition, when available, is stored separately as:
#   vertex<TAB>community


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    paper_name: str
    graph_type: str
    count: int
    directory: str
    graph_names: tuple[str, ...] | None
    requires_truth: bool


@dataclass(frozen=True)
class PaperDatasetRow:
    name: str
    graph_type: str
    nodes: str
    edges: str
    communities: str
    count: int


SBM_SCALES = (
    (1_000, 5_000),
    (5_000, 10_000),
    (10_000, 50_000),
)
SBM_COMMUNITY_COUNTS = (4, 8, 16, 32)
SBM_GRAPHS_PER_STRATUM = 500


def sbm_graph_names() -> tuple[str, ...]:
    return tuple(
        f"SBPGraph_N{nodes}_M{edges}_K{communities}_id{graph_id}"
        for nodes, edges in SBM_SCALES
        for communities in SBM_COMMUNITY_COUNTS
        for graph_id in range(SBM_GRAPHS_PER_STRATUM)
    )


MIT_GRAPH_CHALLENGE_NAMES = tuple(
    f"static_{overlap}Overlap_{variation}BlockSizeVar_{nodes}_nodes"
    for overlap in ("low", "high")
    for variation in ("low", "high")
    for nodes in (1_000_000, 5_000_000)
)

SNAP_GRAPHS = (
    ("SNAP-DBLP", "Undirected"),
    ("SNAP-Youtube", "Undirected"),
    ("SNAP-WikiTopcats", "Directed"),
    ("SNAP-LiveJournal", "Directed"),
    ("SNAP-Orkut", "Undirected"),
)
SNAP_NAMES = tuple(name for name, _graph_type in SNAP_GRAPHS)

PAPER_DATASET_ROWS = (
    PaperDatasetRow(
        "Stochastic Block Model (SBM)",
        "Undirected",
        "1K, 5K, 10K",
        "5K, 10K, 50K",
        "4, 8, 16, 32",
        6_000,
    ),
    PaperDatasetRow(
        "MIT Graph Challenge",
        "Directed",
        "1M, 5M",
        "23.8M, 118.8M",
        "125, 221",
        8,
    ),
    PaperDatasetRow(
        "SNAP-DBLP", "Undirected", "317,080", "1,049,866", "13,477", 1
    ),
    PaperDatasetRow(
        "SNAP-Youtube", "Undirected", "1,134,890", "2,987,624", "8,385", 1
    ),
    PaperDatasetRow(
        "SNAP-WikiTopcats",
        "Directed",
        "1,791,489",
        "28,511,807",
        "17,364",
        1,
    ),
    PaperDatasetRow(
        "SNAP-LiveJournal",
        "Directed",
        "3,997,962",
        "34,681,189",
        "287,512",
        1,
    ),
    PaperDatasetRow(
        "SNAP-Orkut",
        "Undirected",
        "3,072,441",
        "117,185,083",
        "6,288,363",
        1,
    ),
    PaperDatasetRow(
        "CAIDA",
        "Directed",
        "approximately 4,414,370",
        "approximately 16,056,211",
        "N/A",
        30,
    ),
)

DATASETS = {
    "sbm": DatasetSpec(
        key="sbm",
        paper_name="Stochastic Block Model (SBM)",
        graph_type="Undirected",
        count=6_000,
        directory="SBM",
        graph_names=sbm_graph_names(),
        requires_truth=True,
    ),
    "mit-graph-challenge": DatasetSpec(
        key="mit-graph-challenge",
        paper_name="MIT Graph Challenge",
        graph_type="Directed",
        count=8,
        directory="MITGraphChallenge",
        graph_names=MIT_GRAPH_CHALLENGE_NAMES,
        requires_truth=True,
    ),
    "snap": DatasetSpec(
        key="snap",
        paper_name="SNAP",
        graph_type="Paper-specific; see README",
        count=5,
        directory="SNAP",
        graph_names=SNAP_NAMES,
        requires_truth=False,
    ),
    "caida": DatasetSpec(
        key="caida",
        paper_name="CAIDA",
        graph_type="Directed",
        count=30,
        directory="CAIDA",
        graph_names=None,
        requires_truth=False,
    ),
}

CAIDA_GRAPH_NAME = re.compile(r"\d{8}-\d{6}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show", help="Print the paper corpus and local paths.")

    check = subparsers.add_parser("check", help="Validate locally prepared files.")
    check.add_argument("--data-root", type=Path, default=Path("Data"))
    check.add_argument(
        "--dataset",
        choices=("all", *DATASETS),
        default="all",
    )
    check.add_argument(
        "--check-format",
        action="store_true",
        help="Also validate the first data row of every edge list.",
    )
    return parser.parse_args()


def edge_list_names(directory: Path) -> set[str]:
    return {
        path.stem
        for path in directory.glob("*.tsv")
        if not path.stem.endswith("_truePartition")
    }


def first_data_row_error(path: Path) -> str | None:
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", "%")):
                    continue
                fields = stripped.split("\t")
                if len(fields) != 3:
                    return f"{path}:{line_number}: expected three tab-separated fields"
                try:
                    int(fields[0])
                    int(fields[1])
                    weight = float(fields[2])
                except ValueError:
                    return f"{path}:{line_number}: invalid vertex ID or weight"
                if not math.isfinite(weight):
                    return f"{path}:{line_number}: weight must be finite"
                return None
    except OSError as exc:
        return f"{path}: {exc}"
    return f"{path}: no data rows"


def validate_dataset(
    spec: DatasetSpec, data_root: Path, check_format: bool
) -> list[str]:
    directory = data_root / spec.directory
    if not directory.is_dir():
        return [f"missing directory: {directory}"]

    found_names = edge_list_names(directory)
    errors = []
    if spec.graph_names is None:
        invalid = sorted(name for name in found_names if not CAIDA_GRAPH_NAME.fullmatch(name))
        if invalid:
            errors.append(
                "CAIDA filenames must use YYYYMMDD-HHMMSS: " + ", ".join(invalid[:5])
            )
        if len(found_names) != spec.count:
            errors.append(f"expected {spec.count} edge lists, found {len(found_names)}")
        checked_names = found_names
    else:
        expected_names = set(spec.graph_names)
        missing = sorted(expected_names - found_names)
        extra = sorted(found_names - expected_names)
        if missing:
            errors.append(f"missing {len(missing)} edge list(s): {', '.join(missing[:5])}")
        if extra:
            errors.append(f"unexpected edge list(s): {', '.join(extra[:5])}")
        checked_names = expected_names & found_names

    if spec.requires_truth:
        missing_truth = sorted(
            name
            for name in checked_names
            if not (directory / f"{name}_truePartition.tsv").is_file()
        )
        if missing_truth:
            errors.append(
                f"missing {len(missing_truth)} truth partition(s): "
                + ", ".join(missing_truth[:5])
            )

    if check_format:
        for name in sorted(checked_names):
            error = first_data_row_error(directory / f"{name}.tsv")
            if error:
                errors.append(error)
    return errors


def show_manifest() -> None:
    print("Paper dataset corpus")
    print("Dataset | Type | Nodes | Edges | Communities | Graphs")
    for row in PAPER_DATASET_ROWS:
        print(
            f"{row.name} | {row.graph_type} | {row.nodes} | {row.edges} | "
            f"{row.communities} | {row.count:,}"
        )
    print(f"Total graphs: {sum(row.count for row in PAPER_DATASET_ROWS):,}")
    print("Local directories")
    for spec in DATASETS.values():
        print(f"{spec.paper_name} | Data/{spec.directory}/")


def main() -> None:
    args = parse_args()
    if args.command == "show":
        show_manifest()
        return

    selected = DATASETS.values() if args.dataset == "all" else (DATASETS[args.dataset],)
    failed = False
    for spec in selected:
        errors = validate_dataset(spec, args.data_root, args.check_format)
        status = "OK" if not errors else "INCOMPLETE"
        print(f"{spec.paper_name}: {status}")
        for error in errors:
            print(f"  - {error}")
        failed = failed or bool(errors)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
