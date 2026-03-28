#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, List, Optional


def _read_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # pylint: disable=broad-except
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print a specific record from a JSON file by index.\n"
            "Typical usage: inspect one pair in error_fix_pairs.json."
        ),
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to a JSON file (e.g., error_fix_pairs.json).",
    )
    parser.add_argument(
        "--idx",
        type=int,
        required=True,
        help="Index of the record to print (0-based).",
    )

    args = parser.parse_args()
    json_path = os.path.abspath(args.json_path)

    data = _read_json(json_path)
    if not isinstance(data, list):
        print(f"[get_content] ERROR: json_path does not contain a list: {json_path}")
        return 1

    if args.idx < 0 or args.idx >= len(data):
        print(
            f"[get_content] ERROR: idx {args.idx} out of range "
            f"for list of length {len(data)} in {json_path}"
        )
        return 1

    record = data[args.idx]
    # Print as pretty JSON so that the agent or user can easily consume it.
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
