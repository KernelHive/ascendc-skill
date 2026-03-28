#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import difflib


def _read_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # pylint: disable=broad-except
        return None


@dataclass
class TrajRecord:
    idx: int
    code: str
    result: Dict[str, Any]


def _classify_records(traj: List[Dict[str, Any]]) -> Tuple[List[TrajRecord], List[TrajRecord]]:
    """Split traj records into error and correct lists.

    The classification heuristic should be consistent with scripts/get_error_code_num.py:
    - A record is considered "correct" if:
      - result["exit_code"] == 0, AND
      - result["summary"].get("correctness", True) is truthy (if present).
    - Otherwise it is considered "error".
    """
    error_recs: List[TrajRecord] = []
    correct_recs: List[TrajRecord] = []

    for idx, rec in enumerate(traj):
        code = rec.get("code") or ""
        result = rec.get("result") or {}
        if not isinstance(result, dict):
            error_recs.append(TrajRecord(idx=idx, code=code, result={}))
            continue

        exit_code = result.get("exit_code")
        summary = result.get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}
        correctness = summary.get("correctness")

        is_correct = exit_code == 0 and (correctness is True or correctness is None)

        target_list = correct_recs if is_correct else error_recs
        target_list.append(TrajRecord(idx=idx, code=code, result=result))

    return error_recs, correct_recs


def _infer_op_from_result(rec: TrajRecord) -> Optional[str]:
    """Best-effort inference of op name from a traj record."""
    result = rec.result
    if not isinstance(result, dict):
        return None

    # 1. Try summary.op
    summary = result.get("summary") or {}
    if isinstance(summary, dict):
        op = summary.get("op")
        if isinstance(op, str) and op:
            return op

    # 2. Try top-level op
    op = result.get("op")
    if isinstance(op, str) and op:
        return op

    # 3. Try from kernel_code_path filename
    kernel_path = result.get("kernel_code_path")
    if isinstance(kernel_path, str) and kernel_path:
        base = os.path.basename(kernel_path)
        name, _ = os.path.splitext(base)
        if name:
            return name

    return None


def _pair_error_with_correct(
    errors: List[TrajRecord],
    corrects: List[TrajRecord],
) -> List[Tuple[TrajRecord, TrajRecord]]:
    """Pair each error record with a correct record.

    Strategy:
    - Prefer the first correct record whose idx is greater than the error idx
      (i.e., a later successful fix).
    - If none exists, fall back to the first correct record in the list.
    """
    if not corrects:
        return []

    pairs: List[Tuple[TrajRecord, TrajRecord]] = []
    sorted_corrects = sorted(corrects, key=lambda r: r.idx)

    for err in errors:
        later_correct = None
        for c in sorted_corrects:
            if c.idx > err.idx:
                later_correct = c
                break
        if later_correct is None:
            later_correct = sorted_corrects[0]
        pairs.append((err, later_correct))

    return pairs


def _make_unified_diff(error_code: str, correct_code: str, from_name: str, to_name: str) -> str:
    error_lines = error_code.splitlines(keepends=True)
    correct_lines = correct_code.splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(
            error_lines,
            correct_lines,
            fromfile=from_name,
            tofile=to_name,
        )
    )
    return "".join(diff_lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate error-fix code pairs and diffs from a traj.json file.\n"
            "This script reads traj.json produced by scripts/verify.py, pairs "
            "error codes with correct codes, and writes error_fix_pairs.json."
        ),
    )
    parser.add_argument(
        "--traj_path",
        type=str,
        required=True,
        help="Path to traj.json generated alongside result_json by scripts/verify.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="error_fix_pairs.json",
        help=(
            "Output path for error_fix_pairs.json. "
            "Default: ./error_fix_pairs.json in the current working directory."
        ),
    )
    parser.add_argument(
        "--op",
        type=str,
        default=None,
        help=(
            "Optional operator name. If not provided, the script will try to "
            "infer it from traj records (summary.op or kernel_code_path)."
        ),
    )

    args = parser.parse_args()
    traj_path = os.path.abspath(args.traj_path)

    data = _read_json(traj_path)
    if not isinstance(data, list):
        print(f"[get_code_diff] ERROR: traj_path does not contain a list: {traj_path}")
        return 1

    error_recs, correct_recs = _classify_records(data)

    if not error_recs or not correct_recs:
        print(
            "[get_code_diff] WARNING: Need both error and correct records to build pairs. "
            f"error_count={len(error_recs)}, correct_count={len(correct_recs)}"
        )
        return 0

    # Determine op name
    op = args.op
    if not op:
        # Try to infer from the first available record (prefer correct)
        candidate = correct_recs[0] if correct_recs else error_recs[0]
        op = _infer_op_from_result(candidate) or "unknown_op"

    pairs_rc = _pair_error_with_correct(error_recs, correct_recs)

    pairs_json: List[Dict[str, Any]] = []
    for pair_idx, (err, cor) in enumerate(pairs_rc):
        # Best-effort extraction of an error message / reason
        result = err.result or {}
        summary = result.get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}
        reason = summary.get("reason")
        if not isinstance(reason, str) or not reason:
            # Try nested error_message
            inner_result = result.get("result")
            if isinstance(inner_result, list) and inner_result:
                first = inner_result[0]
                if isinstance(first, dict):
                    compile_info = first.get("compile_info") or ""
                    correctness_info = first.get("correctness_info") or ""
                    reason = (compile_info + correctness_info).strip()
            elif isinstance(inner_result, dict):
                compile_info = inner_result.get("compile_info") or ""
                correctness_info = inner_result.get("correctness_info") or ""
                reason = (compile_info + correctness_info).strip()
            if not isinstance(reason, str):
                reason = ""
        code_pair_name = f"{op}_{cor.idx}_{err.idx}"
        diff_text = _make_unified_diff(
            error_code=err.code,
            correct_code=cor.code,
            from_name=f"error_idx_{err.idx}",
            to_name=f"correct_idx_{cor.idx}",
        )

        pair_obj: Dict[str, Any] = {
            "idx": pair_idx,
            "code_pair_name": code_pair_name,
            "op": op,
            "error_idx": err.idx,
            "correct_idx": cor.idx,
            "error_code": err.code,
            "code_diff": diff_text,
            "error": reason,
            "summary": "",
        }
        pairs_json.append(pair_obj)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs_json, f, ensure_ascii=False, indent=2)

    print(
        f"[get_code_diff] Generated {len(pairs_json)} error-fix pairs "
        f"from traj={traj_path} into {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
