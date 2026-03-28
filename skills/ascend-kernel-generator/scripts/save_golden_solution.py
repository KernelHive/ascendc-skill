#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


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
        json.dump(data, f, indent=4, ensure_ascii=False)


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


def _save_golden_solution_from_result(
    op: str,
    verify_result: Any,
    kernel_code_path: str,
    reference_path: Optional[str],
    output_dir: Optional[str] = None,
) -> int:
    """Save golden solution from in-memory verify_result.

    This function extracts the result item from verify_result, which can be:
    - A dict with a "result" field (list[dict]) - new format from verify.py
    - A list[dict] - legacy format
    - A dict - single result item

    Args:
        op: Operator name
        verify_result: Verification result (dict with "result" field, list[dict], or dict)
        kernel_code_path: Path to kernel code file (may be overridden by verify_result)
        reference_path: Optional path to torch reference implementation (may be overridden)
        output_dir: Optional output directory (defaults to skill_root/reference/golden_solutions)

    Returns:
        0 on success, non-zero on failure
    """
    if verify_result is None:
        print("[save_golden_solution] verify_result is None, skip saving.", flush=True)
        return 1

    # Handle new format: dict with "result" field (from verify.py)
    if isinstance(verify_result, dict) and "result" in verify_result:
        # Extract result list from the top-level dict
        result_list = verify_result.get("result")
        if isinstance(result_list, list) and result_list:
            item = result_list[0]
        else:
            print(
                "[save_golden_solution] verify_result has 'result' field but it's not a valid list, "
                "skip saving.",
                flush=True,
            )
            return 1
        # Override kernel_code_path and reference_path from verify_result if available
        if "kernel_code_path" in verify_result:
            kernel_code_path = os.path.abspath(verify_result["kernel_code_path"])
        if "reference_path" in verify_result and not reference_path:
            reference_path = os.path.abspath(verify_result.get("reference_path"))
        # Override op from result item if available
        if isinstance(item, dict) and "op" in item:
            op = item["op"]
    # Handle legacy format: list[dict]
    elif isinstance(verify_result, list) and verify_result:
        item = verify_result[0]
    # Handle single dict format
    else:
        item = verify_result

    if not isinstance(item, dict):
        print("[save_golden_solution] verify_result item is not a dict, skip saving.", flush=True)
        return 1

    compiled = item.get("compiled")
    correctness = item.get("correctness")
    if not compiled or not correctness:
        # 只有在 compiled=True 且 correctness=True 时才保存
        print(
            "[save_golden_solution] Skip saving golden solution because "
            f"compiled={compiled}, correctness={correctness}",
            flush=True,
        )
        return 2

    if not os.path.exists(kernel_code_path):
        print(
            f"[save_golden_solution] Kernel code file not found at {kernel_code_path}, "
            "skip saving golden solution.",
            flush=True,
        )
        return 3

    # 读取 AscendC kernel 描述文件源码
    try:
        with open(kernel_code_path, "r", encoding="utf-8") as f:
            golden_code = f.read()
    except Exception as e:  # noqa: BLE001
        print(
            f"[save_golden_solution] Failed to read kernel code from {kernel_code_path}: {e!r}",
            flush=True,
        )
        return 4

    # 解析 skill_root 并构造 golden_solutions 目录
    skill_root = _get_skill_root()
    if output_dir is None:
        output_dir = os.path.join(skill_root, "reference", "golden_solutions")
    else:
        output_dir = os.path.abspath(output_dir)
    info_path = os.path.join(output_dir, "info.json")
    target_code_path = os.path.join(output_dir, f"{op}.py")

    print(f"[save_golden_solution] Skill root: {skill_root}", flush=True)
    print(f"[save_golden_solution] Operator: {op}", flush=True)
    print(f"[save_golden_solution] Kernel code path: {kernel_code_path}", flush=True)
    print(
        f"[save_golden_solution] Output directory (golden_solutions): {output_dir}",
        flush=True,
    )

    # 加载 / 初始化 info.json
    info = _read_json(info_path) or {}
    if not isinstance(info, dict):
        info = {}

    perf = item.get("performance") or {}
    hardware = item.get("hardware")

    # 计算 reference_path 的相对路径（如果提供）
    rel_ref_path: Optional[str] = None
    if reference_path:
        try:
            rel_ref_path = os.path.relpath(reference_path, start=output_dir)
        except Exception:  # pylint: disable=broad-except
            try:
                rel_ref_path = os.path.relpath(reference_path, start=skill_root)
            except Exception:  # pylint: disable=broad-except
                rel_ref_path = reference_path

    info[op] = {
        **item,
        "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    # Note: ref_path is commented out in verify.py's original implementation
    # if rel_ref_path:
    #     info[op]["ref_path"] = rel_ref_path

    try:
        _write_json(info_path, info)
    except Exception as e:  # noqa: BLE001
        print(
            f"[save_golden_solution] Failed to write info.json at {info_path}: {e!r}",
            flush=True,
        )
        return 5

    try:
        os.makedirs(os.path.dirname(target_code_path), exist_ok=True)
        with open(target_code_path, "w", encoding="utf-8") as f:
            f.write(golden_code)
    except Exception as e:  # noqa: BLE001
        print(
            f"[save_golden_solution] Failed to write golden solution code to "
            f"{target_code_path}: {e!r}",
            flush=True,
        )
        return 6

    print(f"[save_golden_solution] Saved golden solution to {target_code_path}", flush=True)
    print(f"[save_golden_solution] Updated info.json at {info_path}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save a verified AscendC kernel implementation as golden_solution."
    )
    parser.add_argument(
        "--op",
        type=str,
        required=False,
        default=None,
        help=(
            "Operator name, e.g. mse_loss, add. "
            "If not provided, will be extracted from verify_result if available."
        ),
    )
    parser.add_argument(
        "--kernel_code_path",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to kernel code file to save as golden solution. "
            "If not provided, will be extracted from verify_result if available."
        ),
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        default=None,
        help="Optional path to torch reference implementation (for info.json).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory for golden solution. "
            "Default: <skill_root>/reference/golden_solutions, where skill_root is "
            "the install path of this skill (optionally provided via ASCEND_SKILL_ROOT)."
        ),
    )
    # Support two modes: from file or from JSON string
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--result_path",
        type=str,
        help="Path to verification result JSON file (legacy mode).",
    )

    args = parser.parse_args()

    # Convert paths to absolute paths (user-facing paths)
    # Will be overridden by verify_result if available
    kernel_code_path = os.path.abspath(args.kernel_code_path) if args.kernel_code_path else None
    reference_path = os.path.abspath(args.reference_path) if args.reference_path else None

    # Determine default output directory relative to the skill install root
    skill_root = _get_skill_root()
    default_output_dir = os.path.join(skill_root, "reference", "golden_solutions")
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else default_output_dir

    print(f"[save_golden_solution] Skill root: {skill_root}")
    print(f"[save_golden_solution] Operator: {args.op}")
    print(f"[save_golden_solution] Kernel code path: {kernel_code_path}")
    print(f"[save_golden_solution] Output directory (golden_solutions): {output_dir}")

    # Parse verify_result from either file or JSON string

    result_path = os.path.abspath(args.result_path)
    print(f"[save_golden_solution] Result path: {result_path}")
    result = _read_json(result_path)
    if not isinstance(result, (list, dict)):
        print(
            f"[save_golden_solution] ERROR: invalid or missing result.json at {result_path}",
            flush=True,
        )
        return 1
    verify_result = result

    # Extract result item for validation and extract paths/op from verify_result if available
    item = None
    if isinstance(verify_result, dict) and "result" in verify_result:
        # New format: dict with "result" field
        result_list = verify_result.get("result")
        if isinstance(result_list, list) and result_list:
            item = result_list[0]
        # Override paths from verify_result if available
        if "kernel_code_path" in verify_result:
            kernel_code_path = os.path.abspath(verify_result["kernel_code_path"])
        if "reference_path" in verify_result and not reference_path:
            reference_path = os.path.abspath(verify_result["reference_path"])
    elif isinstance(verify_result, list) and verify_result:
        # Legacy format: list[dict]
        item = verify_result[0]
    else:
        # Single dict format
        item = verify_result

    if not isinstance(item, dict):
        print(
            "[save_golden_solution] ERROR: verify_result is not a valid dict or list[dict]",
            flush=True,
        )
        return 1

    # Override op from result item if available, or use from args
    if "op" in item:
        op = item["op"]
    elif args.op:
        op = args.op
    else:
        print(
            "[save_golden_solution] ERROR: op is required but not found in verify_result and "
            "not provided via --op argument.",
            flush=True,
        )
        return 1

    # Validate kernel_code_path is available
    if not kernel_code_path:
        print(
            "[save_golden_solution] ERROR: kernel_code_path is required but not found in "
            "verify_result and not provided via --kernel_code_path argument.",
            flush=True,
        )
        return 1

    compiled = item.get("compiled")
    correctness = item.get("correctness")
    if not compiled or not correctness:
        print(
            "[save_golden_solution] ERROR: compiled or correctness is False/None, "
            "refuse to save as golden solution.",
            flush=True,
        )
        print(f"[save_golden_solution] compiled={compiled}, correctness={correctness}", flush=True)
        return 2

    # Use the unified save function
    return _save_golden_solution_from_result(
        op=op,
        verify_result=verify_result,
        kernel_code_path=kernel_code_path,
        reference_path=reference_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())

