#!/usr/bin/env python3
"""Convert a text or gzip-compressed edge list to the repository TSV format."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import TextIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--unit-weight",
        action="store_true",
        help="Write weight 1 even when the input has a third field.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def convert(input_path: Path, output_path: Path, unit_weight: bool) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        raise SystemExit(f"Temporary output already exists: {temporary_path}")

    rows = 0
    try:
        with open_input(input_path) as source, temporary_path.open(
            "w", encoding="utf-8"
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", "%")):
                    continue
                fields = stripped.split()
                if len(fields) not in (2, 3):
                    raise ValueError(
                        f"{input_path}:{line_number}: expected two or three fields"
                    )
                source_id = int(fields[0])
                destination_id = int(fields[1])
                weight = 1.0 if unit_weight or len(fields) == 2 else float(fields[2])
                destination.write(f"{source_id}\t{destination_id}\t{weight:g}\n")
                rows += 1
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    temporary_path.replace(output_path)
    return rows


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input edge list not found: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {args.output}")
    try:
        rows = convert(args.input, args.output, args.unit_weight)
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Edges written: {rows:,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
