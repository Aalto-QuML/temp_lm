#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path


def merge_csvs(pattern: str, output: str, add_source_file: bool = False) -> Path:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    with output_path.open("w", newline="") as fout:
        for file_path in files:
            with open(file_path, newline="") as fin:
                reader = csv.DictReader(fin)
                if reader.fieldnames is None:
                    continue

                fieldnames = list(reader.fieldnames)
                if add_source_file and "source_file" not in fieldnames:
                    fieldnames.append("source_file")

                if writer is None:
                    writer = csv.DictWriter(fout, fieldnames=fieldnames)
                    writer.writeheader()

                for row in reader:
                    if add_source_file:
                        row["source_file"] = file_path
                    writer.writerow(row)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge many CSV files into one CSV")
    parser.add_argument(
        "--pattern", required=True, help="Glob pattern for input CSV files"
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--add-source-file",
        action="store_true",
        help="Add a source_file column with the origin file path",
    )
    args = parser.parse_args()

    out_path = merge_csvs(
        pattern=args.pattern,
        output=args.out,
        add_source_file=args.add_source_file,
    )
    print(f"Merged CSV written to: {out_path}")


if __name__ == "__main__":
    main()
