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


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Update the 'summary' field for a given pair in error_fix_pairs.json.\n"
            "Agent is expected to first inspect the pair content (e.g. with "
            "scripts/get_content.py), then summarize the fix and call this script "
            "with the corresponding idx and summary."
        ),
    )
    parser.add_argument(
        "--pairs_path",
        type=str,
        default="error_fix_pairs.json",
        help="Path to error_fix_pairs.json (default: ./error_fix_pairs.json).",
    )
    parser.add_argument(
        "--idx",
        type=int,
        required=True,
        help="Index of the pair to update (0-based).",
    )
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Short natural language summary describing how the fix changes the code.",
    )

    args = parser.parse_args()
    pairs_path = os.path.abspath(args.pairs_path)

    data = _read_json(pairs_path)
    if not isinstance(data, list):
        print(
            "[extract_error_fix_into_experience] ERROR: pairs_path does not contain a list: "
            f"{pairs_path}"
        )
        return 1

    if args.idx < 0 or args.idx >= len(data):
        print(
            f"[extract_error_fix_into_experience] ERROR: idx {args.idx} out of range "
            f"for list of length {len(data)} in {pairs_path}"
        )
        return 1

    pairs: List[Any] = data
    pair_obj = pairs[args.idx]
    if not isinstance(pair_obj, dict):
        print(
            f"[extract_error_fix_into_experience] ERROR: item at idx {args.idx} "
            f"is not an object in {pairs_path}"
        )
        return 1

    pair_obj["summary"] = args.summary
    if "idx" in pair_obj:
        pair_obj["idx"] = args.idx

    _write_json(pairs_path, pairs)
    print(
        "[extract_error_fix_into_experience] Updated summary for "
        f"idx={args.idx} in {pairs_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

