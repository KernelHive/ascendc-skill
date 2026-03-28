#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import urllib.request
import urllib.error
import urllib.parse

from filter_hacked_code import filter_code_result_all

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

def _read_code(path: str) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:  # pylint: disable=broad-except
        return ""

# 评估服务地址，可通过环境变量覆盖
# 默认指向非流式 /verify，若使用 --stream，会自动切换到 /verify_stream。
DEFAULT_VERIFY_URL = "http://127.0.0.1:23457/verify"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AscendC kernel HTTP verify client (uses urllib).",
    )
    parser.add_argument(
        "--op",
        required=True,
        help="Operator name, e.g. add, mse_loss.",
    )
    parser.add_argument(
        "--reference_path",
        required=True,
        help="Path to the torch reference implementation file.",
    )
    parser.add_argument(
        "--kernel_code_path",
        required=True,
        help="Path to the AscendC kernel description file.",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated device list, e.g. 'npu:0' or 'npu:0,npu:1'.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("ASCEND_VERIFY_URL", DEFAULT_VERIFY_URL),
        help=f"Verify service URL (default: {DEFAULT_VERIFY_URL}, or $ASCEND_VERIFY_URL).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming API (/verify_stream) to get real-time logs.",
    )
    parser.add_argument(
        "--result_json_path",
        type=str,
        default='./result.json',
        help="Path to save the final JSON result from verify service.",
    )

    args = parser.parse_args()

    # 规范化路径
    reference_path = os.path.abspath(args.reference_path)
    kernel_code_path = os.path.abspath(args.kernel_code_path)

    # 在调用评估服务前，先对 AscendC kernel 描述文件做 hack 级别检查。
    # 如果检测到使用了 Torch 的高阶 API / 预构建算子，则直接返回 compiled=False。
    kernel_code = _read_code(kernel_code_path)
    if kernel_code:
        try:
            ok, reason = filter_code_result_all(kernel_code)
        except Exception as e:  # noqa: BLE001
            print(
                "[verify] Failed to run hack filter, fallback to normal verify. "
                f"error={e!r}. Please do not implement computation logic in `model_src` or `python_bind_src`, or directly call functions from the torch library, such as `at::conv2d`. Instead, implement the computation logic in the `Compute()` function within `kernel_src`, using fully custom-designed kernel functions to implement operators.",
                flush=True,
            )
            ok = True
            reason = ""

        if not ok:
            hack_message = (
                "This code uses Torch's API or pre-built methods, which is not what we want. "
                f"{reason}. Please do not implement computation logic in `model_src` or `python_bind_src`, or directly call functions from the torch library, such as `at::conv2d`. Instead, implement the computation logic in the `Compute()` function within `kernel_src`, using fully custom-designed kernel functions to implement operators."
            )
            print("[verify] Hack-style implementation detected, skip calling verify service. Please do not implement computation logic in `model_src` or `python_bind_src`, or directly call functions from the torch library, such as `at::conv2d`. Instead, implement the computation logic in the `Compute()` function within `kernel_src`, using fully custom-designed kernel functions to implement operators.", flush=True)
            print(hack_message, flush=True)

            # 构造一个与 /verify 返回结构尽量兼容的 JSON，方便上游保存 result_json。
            resp_json = {
                "exit_code": 1,
                "reference_path": reference_path,
                "kernel_code_path": kernel_code_path,
                "result": [
                    {
                        "compiled": False,
                        "correctness": False,
                        "performance": {},
                        "error_message": hack_message,
                    }
                ],
                "summary": {
                    "compiled": False,
                    "correctness": False,
                    "reason": hack_message,
                },
            }

            if args.result_json_path:
                try:
                    json_path = os.path.abspath(args.result_json_path)
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(resp_json, f, ensure_ascii=False, indent=2)
                    print(f"[verify] Final JSON result saved to: {json_path}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[verify] Failed to save result JSON to {args.result_json_path}: {e!r}",
                        flush=True,
                    )

            # compiled=False，直接返回非零 exit code
            return 1

    # 解析设备列表
    devices = None
    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]

    # 构造请求体（与 verify_server.VerifyRequest 一致：基于路径而不是源码字符串）
    payload = {
        "op": args.op,
        "reference_path": reference_path,
        "kernel_code_path": kernel_code_path,
        "language": "ascendc",
        "devices": devices,
    }

    data = json.dumps(payload).encode("utf-8")

    # 打印基础调用信息，方便在长时间等待时观察到进度
    print("[verify] Sending request to verify service...", flush=True)
    print(
        f"[verify] url={args.url}\n"
        f"[verify] op={args.op}\n"
        f"[verify] reference_path={reference_path}\n"
        f"[verify] kernel_code_path={kernel_code_path}\n"
        f"[verify] devices={devices}",
        flush=True,
    )

    # 选择流式或非流式 URL
    use_stream = args.stream
    if use_stream:
        if args.url.endswith("/verify"):
            target_url = args.url[: -len("/verify")] + "/verify_stream"
        else:
            # 用户传了自定义地址，默认直接拼接 /verify_stream
            target_url = args.url.rstrip("/") + "/verify_stream"
        print(f"[verify] Using streaming API: {target_url}", flush=True)
    else:
        # 如果用户把 ASCEND_VERIFY_URL / --url 显式设成了 /verify_stream，
        # 即使没传 --stream 也自动走流式逻辑，避免 JSONDecodeError。
        if args.url.endswith("/verify_stream"):
            use_stream = True
            target_url = args.url
            print(
                "[verify] Detected verify_stream endpoint without --stream; "
                "enabling streaming mode automatically.",
                flush=True,
            )
        else:
            target_url = args.url

    req = urllib.request.Request(
        url=target_url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
        },
    )

    try:
        if use_stream:
            # 流式模式：逐行读取 NDJSON
            print(
                "[verify] Waiting for streaming response from verify service (timeout=600s)...",
                flush=True,
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                exit_code: int | None = None
                final_json: dict | None = None
                for raw_line in resp:
                    try:
                        line = raw_line.decode("utf-8").strip()
                    except UnicodeDecodeError:
                        # 非 UTF-8 内容直接跳过
                        continue
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        # 防御性：服务端如果偶尔输出非 JSON 行，直接原样打印
                        print(line, flush=True)
                        continue

                    msg_type = msg.get("type")
                    if msg_type == "meta":
                        # 基本元信息
                        devices_text = msg.get("devices")
                        print(
                            "[verify-meta]",
                            f"op={msg.get('op')}",
                            f"reference_path={msg.get('reference_path')}",
                            f"kernel_code_path={msg.get('kernel_code_path')}",
                            f"devices={devices_text}",
                            f"language={msg.get('language')}",
                            flush=True,
                        )
                        print(
                            "[verify] Current task is using device(s):",
                            devices_text if devices_text else "default device",
                            flush=True,
                        )
                    elif msg_type == "log":
                        # 评测过程日志：尽量保持原样输出
                        data = msg.get("data", "")
                        # 避免额外换行；原始日志中通常自带 \n
                        sys.stdout.write(str(data))
                        sys.stdout.flush()
                    elif msg_type == "result":
                        print("\n[verify] Final result received:", flush=True)
                        print("exit_code:", msg.get("exit_code"), flush=True)
                        print("summary:", msg.get("summary"), flush=True)
                        # 规范化最终 JSON 结构，便于与 /verify 一致使用
                        final_json = {
                            "exit_code": msg.get("exit_code"),
                            "reference_path": msg.get("reference_path"),
                            "kernel_code_path": msg.get("kernel_code_path"),
                            "result": msg.get("result"),
                            "summary": msg.get("summary"),
                        }
                        # 如需查看详细结果，可自行取消下注释：
                        # print("full result:", json.dumps(msg.get("result"), indent=2, ensure_ascii=False))
                        exit_code = int(msg.get("exit_code", 1))
                    elif msg_type == "error":
                        print("[verify] ERROR from server:", msg.get("message"), flush=True)
                        exit_code = 1
                    else:
                        # 未知类型，安全起见直接打印
                        print(line, flush=True)

                if exit_code is None:
                    print(
                        "[verify] Streaming finished without 'result' message, treating as failure.",
                        flush=True,
                    )
                    return 1
                # 如用户指定了结果输出路径，则保存最终 JSON
                if args.result_json_path and final_json is not None:
                    try:
                        json_path = os.path.abspath(args.result_json_path)
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(final_json, f, ensure_ascii=False, indent=2)
                        print(f"[verify] Final JSON result saved to: {json_path}", flush=True)

                        traj_path = os.path.join(os.path.dirname(json_path), "traj.json")
                        if os.path.exists(traj_path):
                            with open(traj_path, "r", encoding="utf-8") as f:
                                traj = json.load(f)
                        else:
                            traj = []

                        traj.append({
                            "idx": len(traj),
                            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
                            "code": _read_code(kernel_code_path),
                            "result": final_json,
                        })
                        with open(traj_path, "w", encoding="utf-8") as f:
                            json.dump(traj, f, ensure_ascii=False, indent=2)
                        print(f"[verify] Trajectory saved to: {traj_path}", flush=True)

                    except Exception as e:  # noqa: BLE001
                        print(
                            f"[verify] Failed to save result JSON to {args.result_json_path}: {e!r}",
                            flush=True,
                        )

                return exit_code
        else:
            # 非流式模式：一次性拿到 JSON
            print(
                "[verify] Waiting for response from verify service (timeout=600s)...",
                flush=True,
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                resp_bytes = resp.read()
                resp_text = resp_bytes.decode("utf-8")
                resp_json = json.loads(resp_text)

            print("[verify] Response received. Summary:", flush=True)
            print("exit_code:", resp_json.get("exit_code"))
            print("reference_path:", resp_json.get("reference_path"))
            print("kernel_code_path:", resp_json.get("kernel_code_path"))
            print("summary:", resp_json.get("summary"))
            # 如需查看详细结果：
            # print("full result:", json.dumps(resp_json.get("result"), indent=2, ensure_ascii=False))

            # 如用户指定了结果输出路径，则保存最终 JSON
            if args.result_json_path:
                try:
                    json_path = os.path.abspath(args.result_json_path)
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(resp_json, f, ensure_ascii=False, indent=2)
                    print(f"[verify] Final JSON result saved to: {json_path}", flush=True)
                    traj_path = os.path.join(os.path.dirname(json_path), "traj.json")
                    if os.path.exists(traj_path):
                        with open(traj_path, "r", encoding="utf-8") as f:
                            traj = json.load(f)
                    else:
                        traj = []

                    traj.append({
                        "idx": len(traj),
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
                        "code": _read_code(kernel_code_path),
                        "result": resp_json,
                    })
                    with open(traj_path, "w", encoding="utf-8") as f:
                        json.dump(traj, f, ensure_ascii=False, indent=2)
                    print(f"[verify] Trajectory saved to: {traj_path}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[verify] Failed to save result JSON to {args.result_json_path}: {e!r}",
                        flush=True,
                    )

            exit_code = int(resp_json.get("exit_code", 1))
            return exit_code

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print("HTTPError:", e.code, e.reason)
        print("Response body:", body)
        return 1
    except urllib.error.URLError as e:
        print("URLError:", e.reason)
        return 1
    except Exception as e:  # noqa: BLE001
        print("Unexpected error:", repr(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
