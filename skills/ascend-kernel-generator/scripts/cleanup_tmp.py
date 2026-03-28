#!/usr/bin/env python3
import argparse
import os
import shutil
import sys


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
        description="Clean MultiKernelBench tmp directory under this skill workspace."
    )
    parser.add_argument(
        "--skill_root",
        type=str,
        default=None,
        help=(
            "Optional explicit skill_root. If not provided, resolve by ASCEND_SKILL_ROOT "
            "or infer from scripts/ parent directory."
        ),
    )
    parser.add_argument(
        "--op",
        type=str,
        default=None,
        help=(
            "Operator name to clean (e.g., 'tanh_custom' or 'tanh'). "
            "If provided, only folders matching this operator will be cleaned. "
            "If not provided, all operator folders in tmp/ will be cleaned."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print what would be deleted, without actually deleting.",
    )
    parser.add_argument(
        "--ignore_missing",
        action="store_true",
        help="Return success even if tmp directory or operator folders do not exist.",
    )
    args = parser.parse_args()

    skill_root = os.path.abspath(args.skill_root) if args.skill_root else _get_skill_root()
    tmp_dir = os.path.join(skill_root, "scripts", "multi-kernel-bench", "tmp")
    tmp_dir = os.path.abspath(tmp_dir)

    print(f"[cleanup_tmp] Skill root: {skill_root}")
    print(f"[cleanup_tmp] Target tmp dir: {tmp_dir}")

    if not os.path.exists(tmp_dir):
        msg = f"[cleanup_tmp] tmp dir not found: {tmp_dir}"
        if args.ignore_missing:
            print(msg)
            return 0
        print(msg)
        return 2

    if not os.path.isdir(tmp_dir):
        print(f"[cleanup_tmp] ERROR: target exists but is not a directory: {tmp_dir}")
        return 3

    # Find operator folders to clean
    op_folders = []
    if args.op:
        # Normalize operator name: add '_custom' suffix if not present
        op_name = args.op if args.op.endswith("_custom") else f"{args.op}_custom"
        print(f"[cleanup_tmp] Looking for operator folders matching: {op_name}")
        
        # List all directories in tmp_dir
        try:
            for item in os.listdir(tmp_dir):
                item_path = os.path.join(tmp_dir, item)
                if os.path.isdir(item_path) and item.startswith(op_name):
                    op_folders.append(item_path)
        except Exception as e:  # noqa: BLE001
            print(f"[cleanup_tmp] ERROR: failed to list tmp directory: {e!r}")
            return 5
    else:
        # If no --op specified, clean all folders in tmp/
        print("[cleanup_tmp] No operator specified, will clean all folders in tmp/")
        try:
            for item in os.listdir(tmp_dir):
                item_path = os.path.join(tmp_dir, item)
                if os.path.isdir(item_path):
                    op_folders.append(item_path)
        except Exception as e:  # noqa: BLE001
            print(f"[cleanup_tmp] ERROR: failed to list tmp directory: {e!r}")
            return 5

    if not op_folders:
        msg = f"[cleanup_tmp] No operator folders found"
        if args.op:
            msg += f" matching '{args.op}'"
        if args.ignore_missing:
            print(msg)
            return 0
        print(msg)
        return 2

    print(f"[cleanup_tmp] Found {len(op_folders)} folder(s) to clean:")
    for folder in op_folders:
        print(f"  - {folder}")

    if args.dry_run:
        print("[cleanup_tmp] dry-run enabled, skip deleting.")
        return 0

    # Remove each operator folder
    removed_count = 0
    for folder in op_folders:
        try:
            shutil.rmtree(folder)
            print(f"[cleanup_tmp] Removed directory: {folder}")
            removed_count += 1
        except Exception as e:  # noqa: BLE001
            print(f"[cleanup_tmp] ERROR: failed to remove {folder}: {e!r}")
            return 4

    print(f"[cleanup_tmp] Successfully removed {removed_count} folder(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

