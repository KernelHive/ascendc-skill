import argparse
import json
import os
import sys
import textwrap
from typing import Any, Optional, Tuple
import datetime
from datetime import timezone

# 将 multi-kernel-bench 根目录加入 sys.path，方便导入 envs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from envs.env import Env

def _read_code(path: str) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:  # pylint: disable=broad-except
        return ""

def _run_verify(
    op: str,
    reference_path: str,
    kernel_code_path: str,
    language: str = "ascendc",
    devices: Optional[list] = None,
) -> Tuple[int, Any]:
    """Run verification using Env class directly.

    Args:
        op: Operator name
        reference_path: Path to the torch reference implementation file
        kernel_code_path: Path to the kernel code file to verify
        language: Language/platform (default: "ascendc")
        devices: Optional list of devices to use

    Returns:
        (exit_code, result): exit_code is 0 on success, non-zero on failure.
                             result is the Env.step result (list of dicts).
    """
    # Check if files exist
    if not os.path.exists(reference_path):
        return 1, [
            {
                "compiled": False,
                "correctness": False,
                "correctness_info": f"Reference file not found at {reference_path}",
            }
        ]

    if not os.path.exists(kernel_code_path):
        return 1, [
            {
                "compiled": False,
                "correctness": False,
                "correctness_info": f"Kernel code file not found at {kernel_code_path}",
            }
        ]

    # Read kernel code
    try:
        with open(kernel_code_path, "r", encoding="utf-8") as f:
            kernel_code = f.read()
    except Exception as e:
        return 1, [
            {
                "compiled": False,
                "correctness": False,
                "correctness_info": f"Failed to read kernel code from {kernel_code_path}: {e}",
            }
        ]

    # Initialize Env
    try:
        env = Env(
            op=op,
            category="custom",  # Default category, not used for verification
            language=language,
            ref_src_path=reference_path,
            max_workers=10,
            devices=devices,
        )
    except Exception as e:
        return 1, [
            {
                "compiled": False,
                "correctness": False,
                "correctness_info": f"Failed to initialize Env: {e}",
            }
        ]

    # Run verification
    try:
        # Env.step expects a list of code strings
        results = env.step([kernel_code])

        # Check if verification was successful
        if results and isinstance(results, list) and len(results) > 0:
            result = results[0]
            if result.get("compiled") and result.get("correctness"):
                return 0, results
            else:
                return 1, results
        else:
            return 1, results or [
                {
                    "compiled": False,
                    "correctness": False,
                    "correctness_info": "Env.step returned empty or invalid result",
                }
            ]
    except Exception as e:
        import traceback

        return 1, [
            {
                "compiled": False,
                "correctness": False,
                "correctness_info": f"Exception during verification: {e}\n{traceback.format_exc()}",
            }
        ]


def summarize_result(result: Any) -> str:
    """Create a short human-readable summary from Env.step result."""
    if not result:
        return "No result or empty result."

    # Env.step in multi-kernel-bench returns a list of dicts
    if isinstance(result, list) and result:
        item = result[0]
    elif isinstance(result, dict):
        item = result
    else:
        return f"Unexpected result format: {type(result)}"

    compiled = item.get("compiled")
    correctness = item.get("correctness")
    perf = item.get("performance") or {}
    hardware = item.get("hardware")
    correctness_info = item.get("correctness_info", "")

    lines = [
        f"compiled={compiled}, correctness={correctness}, hardware={hardware}",
    ]
    if correctness_info:
        lines.append(f"correctness_info: {correctness_info}")
    if perf:
        lines.append(
            "performance: "
            + ", ".join(
                f"{k}={v}"
                for k, v in perf.items()
                if k in ("mean", "std", "min", "max", "num_trials")
            )
        )
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify AscendC kernel code against torch reference implementation.",
    )
    parser.add_argument(
        "--op",
        type=str,
        required=True,
        help="Operator name (e.g., mse_loss, add).",
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        required=True,
        help="Path to the torch reference implementation file.",
    )
    parser.add_argument(
        "--kernel_code_path",
        type=str,
        required=True,
        help="Path to the kernel code file to verify.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ascendc",
        help="Language/platform (default: ascendc).",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated list of devices (e.g., 'npu:0,npu:1'). If not provided, Env will use default.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save verification result JSON. If not provided, result is only printed.",
    )

    args = parser.parse_args(argv)

    # Convert paths to absolute paths
    reference_path = os.path.abspath(args.reference_path)
    kernel_code_path = os.path.abspath(args.kernel_code_path)

    # 在运行评测服务前先检查文件是否存在，不存在则抛出异常
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference file not found at {reference_path}")
    if not os.path.exists(kernel_code_path):
        raise FileNotFoundError(f"Kernel code file not found at {kernel_code_path}")

    print(f"[verify.py] Operator: {args.op}")
    print(f"[verify.py] Reference path: {reference_path}")
    print(f"[verify.py] Kernel code path: {kernel_code_path}")
    print(f"[verify.py] Language: {args.language}")

    # Parse devices if provided
    devices = None
    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
        print(f"[verify.py] Using devices: {devices}")

    print("[verify.py] Starting verification...")

    exit_code, result = _run_verify(
        op=args.op,
        reference_path=reference_path,
        kernel_code_path=kernel_code_path,
        language=args.language,
        devices=devices,
    )

    # Save result to file if output path is provided
    if args.output:
        try:
            output_path = os.path.abspath(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[verify.py] Result saved to {output_path}")

            traj_path = os.path.join(os.path.dirname(output_path), "traj.json")
            if os.path.exists(traj_path):
                with open(traj_path, "r", encoding="utf-8") as f:
                    traj = json.load(f)
            else:
                traj = []

            traj.append({
                "idx": len(traj),
                "timestamp": datetime.datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
                "code": _read_code(kernel_code_path),
                "result": result,
            })
            with open(traj_path, "w", encoding="utf-8") as f:
                json.dump(traj, f, ensure_ascii=False, indent=2)
            print(f"[verify] Trajectory saved to: {traj_path}", flush=True)
        except Exception as e:
            print(f"[verify.py] WARNING: Failed to save result to {args.output}: {e}")

    # Print summary
    if result:
        print("[verify.py] Verification result summary:")
        print(textwrap.indent(summarize_result(result), prefix="  "))
    else:
        print("[verify.py] WARNING: No result returned from verification.")

    if exit_code != 0:
        print(f"[verify.py] Verification failed, exit code={exit_code}.")
        return exit_code

    print("[verify.py] Verification process finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

