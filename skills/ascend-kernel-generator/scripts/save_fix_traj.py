#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Optional


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


def _get_skill_root() -> str:
    """Return the install root of this skill.

    Resolution order:
    1. Environment variable ASCEND_SKILL_ROOT (if set by the agent / host).
    2. Parent directory of this scripts/ folder (default layout).
    """
    env_root = os.environ.get("ASCEND_SKILL_ROOT")
    if env_root:
        return os.path.abspath(env_root)

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(_script_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize error_fix_pairs.json into skill_root/reference/issues/{op}.json.\n"
            "The output keeps, for each pair, only the fields: {op, error, code_diff, summary}."
        ),
    )
    parser.add_argument(
        "--op",
        type=str,
        required=True,
        help="Operator name (e.g., add, mse_loss). Used for the output filename {op}.json.",
    )
    parser.add_argument(
        "--pairs_path",
        type=str,
        default="error_fix_pairs.json",
        help="Path to error_fix_pairs.json (default: ./error_fix_pairs.json).",
    )

    args = parser.parse_args()
    pairs_path = os.path.abspath(args.pairs_path)

    pairs_data = _read_json(pairs_path)
    if not isinstance(pairs_data, list):
        print(f"[save_fix_traj] ERROR: pairs_path does not contain a list: {pairs_path}")
        return 1

    # Build minimal records
    summarized: List[Dict[str, Any]] = []
    for item in pairs_data:
        if not isinstance(item, dict):
            continue
        record = {
            "op": args.op,
            "error": item.get("error", ""),
            "code_diff": item.get("code_diff", ""),
            "summary": item.get("summary", ""),
        }
        summarized.append(record)

    skill_root = _get_skill_root()
    issues_dir = os.path.join(skill_root, "reference", "issues")
    output_path = os.path.join(issues_dir, f"{args.op}.json")

    _write_json(output_path, summarized)
    print(
        f"[save_fix_traj] Saved {len(summarized)} summarized error/fix records "
        f"for op={args.op} to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

