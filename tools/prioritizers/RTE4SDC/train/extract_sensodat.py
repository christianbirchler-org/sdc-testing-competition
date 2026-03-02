"""Extract road points, test outcome, and test duration from SensoDat XODR files.

Produces a JSON file with the same schema as sdc-test-data.json so that
RTE4SDC can consume it without any parser changes.

Usage:
    python extract_sensodat.py --sensodat-dir data --output data/sensodat.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import xmltodict

EXPECTED_ROAD_POINTS = 197

def parse_xodr(xodr_path: Path) -> dict | None:
    """Parse a single XODR file and return a dict matching sdc-test-data.json schema."""
    with open(xodr_path, "r", encoding="utf-8") as f:
        doc = xmltodict.parse(f.read())

    header = doc["OpenDRIVE"]["header"]
    info = header.get("sdc_test_info")
    if info is None:
        return None

    outcome = info.get("@test_outcome")
    duration = info.get("@test_duration")
    is_valid = info.get("@is_valid", "False")

    if outcome is None or duration is None or duration == "None":
        return None
    if is_valid.lower() != "true":
        return None

    road = doc["OpenDRIVE"]["road"]
    geometry = road["planView"]["geometry"]
    if not isinstance(geometry, list):
        geometry = [geometry]

    road_points = [{"x": float(pt["@x"]), "y": float(pt["@y"])} for pt in geometry]

    if len(road_points) != EXPECTED_ROAD_POINTS:
        return None

    test_id = info.get("@test_id", xodr_path.stem)

    return {
        "_id": {"$oid": test_id},
        "road_points": road_points,
        "meta_data": {
            "test_info": {
                "test_outcome": outcome,
                "test_duration": float(duration),
            }
        },
    }


def collect_xodr_files(sensodat_dir: Path) -> list[Path]:
    """Recursively find all XODR files in sensodat data directory."""
    pattern = re.compile(r"\d{5}-test\.xodr$")
    xodr_files = []
    for root, _, files in os.walk(sensodat_dir):
        for fname in files:
            if pattern.match(fname):
                xodr_files.append(Path(root) / fname)
    return sorted(xodr_files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensodat-dir",
        default="data",
        help="Path to directory containing unzipped SensoDat XODR files",
    )
    parser.add_argument(
        "--output",
        default="data/sensodat.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    sensodat_dir = Path(args.sensodat_dir)
    if not sensodat_dir.exists():
        print(f"Error: {sensodat_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {sensodat_dir} for XODR files...")
    xodr_files = collect_xodr_files(sensodat_dir)
    print(f"Found {len(xodr_files)} XODR files")

    results = []
    skipped = 0
    for i, xodr_path in enumerate(xodr_files):
        entry = parse_xodr(xodr_path)
        if entry is None:
            skipped += 1
            continue
        results.append(entry)
        if (i + 1) % 5000 == 0:
            print(f"  processed {i + 1}/{len(xodr_files)} files ({len(results)} valid)")

    print(f"Done: {len(results)} valid test cases, {skipped} skipped")

    fail_count = sum(1 for r in results if r["meta_data"]["test_info"]["test_outcome"] == "FAIL")
    pass_count = len(results) - fail_count
    print(f"  FAIL: {fail_count}, PASS: {pass_count} ({100 * fail_count / len(results):.1f}% fail rate)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"Saved to {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
