import os
import sys
import json
import threading
import queue
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import tempfile
from contextlib import contextmanager
from typing import List, Optional, Any, Iterable, Callable

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# Ensure we can import local verify helper and multi-kernel-bench stack
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from verify import summarize_result  # type: ignore


def _normalize_device_name(device: Any) -> str:
    if device is None:
        return ""
    text = str(device).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("npu:"):
        suffix = lowered.split(":", 1)[1].strip()
        if suffix.isdigit():
            return f"npu:{int(suffix)}"
        return lowered
    if text.isdigit():
        return f"npu:{int(text)}"
    return text


def _ensure_skill_root_env() -> str:
    """
    Ensure ASCEND_SKILL_ROOT is set for downstream scripts.

    Returns the resolved skill_root path.
    """
    skill_root = os.environ.get("ASCEND_SKILL_ROOT")
    if not skill_root:
        skill_root = REPO_ROOT
        os.environ["ASCEND_SKILL_ROOT"] = skill_root
    return skill_root


def _parse_device_list(text: Optional[str]) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p]


_DEVICE_POOL = _parse_device_list(os.environ.get("ASCEND_VERIFY_DEVICE_POOL"))

# Global inflight limiter: defaults to the number of configured NPU cards.
_MAX_INFLIGHT = int(os.environ.get("ASCEND_VERIFY_MAX_INFLIGHT", "0") or "0")
if _MAX_INFLIGHT <= 0:
    # 默认并行度与可用 NPU 卡数量一致；如果没有配置设备池，则退化为 4。
    _MAX_INFLIGHT = len(_DEVICE_POOL) if _DEVICE_POOL else 4


def _normalize_devices(devices: Optional[List[str]]) -> List[str]:
    if not devices:
        return []
    out: List[str] = []
    for d in devices:
        ds = _normalize_device_name(d)
        if ds:
            out.append(ds)
    return out


@dataclass
class _ScheduledTask:
    requested_devices: List[str]
    on_wait: Optional[Callable[[int], None]] = None
    assigned_devices: List[str] = field(default_factory=list)
    admitted: bool = False
    last_reported_ahead: Optional[int] = None


class _VerifyScheduler:
    """
    Global verify scheduler.

    It maintains a FIFO waiting queue, keeps the number of active verify jobs
    bounded by `_MAX_INFLIGHT`, and only assigns a device when a runnable slot
    is actually available.
    """

    def __init__(self, max_inflight: int, device_pool: List[str]) -> None:
        self._max_inflight = max(1, max_inflight)
        self._device_pool = list(device_pool)
        self._cond = threading.Condition()
        self._pending: List[_ScheduledTask] = []
        self._running = 0
        self._busy_devices: set[str] = set()
        self._rr = 0

    def _pick_any_free_device_locked(self) -> List[str]:
        if not self._device_pool:
            return []
        pool_size = len(self._device_pool)
        for offset in range(pool_size):
            idx = (self._rr + offset) % pool_size
            dev = self._device_pool[idx]
            if dev not in self._busy_devices:
                self._rr = (idx + 1) % pool_size
                return [dev]
        return []

    def _try_resolve_devices_locked(self, task: _ScheduledTask) -> Optional[List[str]]:
        requested = sorted(set(_normalize_devices(task.requested_devices)))
        if requested:
            if any(dev in self._busy_devices for dev in requested):
                return None
            return requested
        if self._device_pool:
            return self._pick_any_free_device_locked() or None
        return []

    def _dispatch_locked(self) -> None:
        while self._running < self._max_inflight:
            candidate_index: Optional[int] = None
            candidate_devices: Optional[List[str]] = None

            for idx, task in enumerate(self._pending):
                resolved_devices = self._try_resolve_devices_locked(task)
                if resolved_devices is None:
                    continue
                candidate_index = idx
                candidate_devices = resolved_devices
                break

            if candidate_index is None or candidate_devices is None:
                return

            task = self._pending.pop(candidate_index)
            task.assigned_devices = candidate_devices
            task.admitted = True
            self._running += 1
            self._busy_devices.update(candidate_devices)

    @contextmanager
    def acquire(
        self,
        devices: Optional[List[str]],
        on_wait: Optional[Callable[[int], None]] = None,
    ) -> Any:
        task = _ScheduledTask(requested_devices=_normalize_devices(devices), on_wait=on_wait)

        while True:
            notify_ahead: Optional[int] = None
            with self._cond:
                if task not in self._pending and not task.admitted:
                    self._pending.append(task)
                    self._dispatch_locked()
                    self._cond.notify_all()

                if task.admitted:
                    assigned_devices = list(task.assigned_devices)
                    break

                ahead = self._pending.index(task)
                if ahead != task.last_reported_ahead:
                    task.last_reported_ahead = ahead
                    notify_ahead = ahead
                self._cond.wait()

            if notify_ahead is not None and task.on_wait is not None:
                task.on_wait(notify_ahead)

        try:
            yield assigned_devices
        finally:
            with self._cond:
                self._running = max(0, self._running - 1)
                for dev in assigned_devices:
                    self._busy_devices.discard(dev)
                self._dispatch_locked()
                self._cond.notify_all()


_VERIFY_SCHEDULER = _VerifyScheduler(_MAX_INFLIGHT, _DEVICE_POOL)


def _configure_scheduler(device_pool: List[str], max_inflight: int) -> None:
    global _DEVICE_POOL, _MAX_INFLIGHT, _VERIFY_SCHEDULER

    _DEVICE_POOL = list(device_pool)
    resolved_max_inflight = max_inflight
    if resolved_max_inflight <= 0:
        resolved_max_inflight = len(_DEVICE_POOL) if _DEVICE_POOL else 4
    _MAX_INFLIGHT = resolved_max_inflight
    _VERIFY_SCHEDULER = _VerifyScheduler(_MAX_INFLIGHT, _DEVICE_POOL)


def _verify_py_path() -> str:
    """
    Path to multi-kernel-bench/verify.py (CLI entry).
    We run it in a subprocess to isolate sys.stdout/sys.stderr and global state.
    """
    return os.path.join(SCRIPT_DIR, "verify.py")


def _run_verify_isolated(
    *,
    op: str,
    reference_path: str,
    kernel_code_path: str,
    language: str,
    devices: Optional[List[str]],
    stream: bool,
):
    """
    Run verify in a dedicated subprocess.

    Returns:
      - if stream=False: (exit_code, result, logs)
      - if stream=True: (proc, json_path, assigned_devices)
    """
    assigned_devices = _normalize_devices(devices)
    # Scheduling limits: cap overall concurrency and avoid device oversubscription.
    # Note: this is used by /verify and /verify_stream under explicit locks too.
    # For direct calls (e.g. /verify_batch), the caller may or may not wrap it.

    # Output JSON path (verify.py writes Env.step result only)
    tmp = tempfile.NamedTemporaryFile(prefix="verify_result_", suffix=".json", delete=False)
    tmp.close()
    json_path = tmp.name

    cmd = [
        sys.executable,
        _verify_py_path(),
        "--op",
        op,
        "--reference_path",
        reference_path,
        "--kernel_code_path",
        kernel_code_path,
        "--language",
        language,
        "--output",
        json_path,
    ]
    if assigned_devices:
        cmd.extend(["--devices", ",".join(assigned_devices)])

    env = os.environ.copy()
    _ensure_skill_root_env()

    if stream:
        proc = subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        return proc, json_path, assigned_devices

    cp = subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        env=env,
    )
    logs = (cp.stdout or "") + (cp.stderr or "")
    result: Any = None
    try:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                result = json.load(f)
    except Exception as e:  # noqa: BLE001
        result = [
            {
                "compiled": False,
                "correctness": False,
                "correctness_info": f"Failed to read verify output JSON: {e!r}",
            }
        ]
    finally:
        try:
            os.remove(json_path)
        except OSError:
            pass

    return int(cp.returncode), result, logs, assigned_devices


def _run_verify_isolated_scheduled(
    *,
    op: str,
    reference_path: str,
    kernel_code_path: str,
    language: str,
    devices: Optional[List[str]],
) -> tuple[int, Any, str, List[str]]:
    """
    Non-stream scheduled runner:
      - enqueue verify task into the global scheduler
      - keep runtime concurrency aligned with NPU card count
      - run verify in a subprocess and collect (exit_code, result, logs)

    When this task is waiting in the queue, every queue-position update is
    collected into logs so the caller can inspect the full wait history.
    """
    waiting_messages: List[str] = []

    def _collect_waiting_message(ahead: int) -> None:
        waiting_messages.append(
            (
                "Waiting for testing, there are "
                f"{ahead} more operators preceding this one that are awaiting evaluation.\n"
            )
        )

    with _VERIFY_SCHEDULER.acquire(devices, _collect_waiting_message) as scheduled_devices:
        waiting_messages.append(
            f"Current task is using device(s): {', '.join(scheduled_devices) if scheduled_devices else 'default device'}\n"
        )
        exit_code, result, logs, assigned_devices = _run_verify_isolated(
            op=op,
            reference_path=reference_path,
            kernel_code_path=kernel_code_path,
            language=language,
            devices=scheduled_devices,
            stream=False,
        )

    full_logs = "".join(waiting_messages) + (logs or "")
    return int(exit_code), result, str(full_logs), assigned_devices


class VerifyRequest(BaseModel):
    op: str
    reference_path: str
    kernel_code_path: str
    language: str = "ascendc"
    devices: Optional[List[str]] = None


class VerifyBatchItem(BaseModel):
    """单个算子的批量验证请求条目。"""

    op: str
    reference_path: str
    kernel_code_path: str
    language: str = "ascendc"
    devices: Optional[List[str]] = None


class VerifyBatchResponseItem(BaseModel):
    """批量并行验证中单个条目的返回结果。"""

    op: str
    reference_path: str
    kernel_code_path: str
    exit_code: int
    result: Any
    summary: str


class VerifyBatchRequest(BaseModel):
    """批量并行验证请求，支持一次提交多个 AscendC kernel 进行并行编译与测试。"""

    items: List[VerifyBatchItem]


class VerifyResponse(BaseModel):
    exit_code: int
    reference_path: str
    kernel_code_path: str
    result: Any  # Env.step result (list of dicts)
    summary: str
    logs: str  # Captured stdout/stderr during verification


app = FastAPI(title="Ascend Kernel Verify Service")


@app.post("/verify", response_model=VerifyResponse)
def verify_endpoint(req: VerifyRequest) -> VerifyResponse:
    """
    HTTP 端口服务版本的 AscendC kernel 评测回路（基于文件路径）。

    与 MCP 工具 `verify_kernel` 语义类似：
      - 直接使用传入的参考实现路径与 AscendC 描述文件路径
      - 调用 `_run_verify` 使用 multi-kernel-bench 的 Env 完成编译 + 正确性 + 性能验证
      - 返回 exit_code、原始 result 以及人类可读 summary
    """
    _ensure_skill_root_env()

    reference_path = os.path.abspath(req.reference_path)
    kernel_code_path = os.path.abspath(req.kernel_code_path)

    exit_code, result, logs, _assigned_devices = _run_verify_isolated_scheduled(
        op=req.op,
        reference_path=reference_path,
        kernel_code_path=kernel_code_path,
        language=req.language,
        devices=req.devices,
    )

    summary = summarize_result(result)

    return VerifyResponse(
        exit_code=exit_code,
        reference_path=reference_path,
        kernel_code_path=kernel_code_path,
        result=result,
        summary=summary,
        logs=logs,
    )


@app.post("/verify_stream")
def verify_stream_endpoint(req: VerifyRequest) -> StreamingResponse:
    """
    流式版本的 AscendC kernel 评测接口。

    协议约定：
      - 响应为按行分隔的 JSON（NDJSON），Content-Type: application/x-ndjson
      - 每行是一个 JSON 对象，字段：
          {"type": "log", "data": "<任意文本 chunk>"}           # 评测过程日志（stdout/stderr）
          {"type": "result", "exit_code": int,
           "summary": str, "result": <Env.step 原始结果>}      # 最终结果
          {"type": "error", "message": str}                    # 服务端异常
    客户端可以一边读一边打印 "log" 事件，等待 "result" 再决定返回码。
    """
    _ensure_skill_root_env()

    reference_path = os.path.abspath(req.reference_path)
    kernel_code_path = os.path.abspath(req.kernel_code_path)

    q: "queue.Queue[object]" = queue.Queue()
    sentinel = object()

    def worker() -> None:
        try:
            def _notify_wait(ahead: int) -> None:
                """
                当当前算子需要等待评测队列时，向客户端输出一条日志：
                "Waiting for testing, there are {ahead} more operators preceding this one that are awaiting evaluation."
                """
                q.put(
                    json.dumps(
                        {
                            "type": "log",
                            "data": (
                                "Waiting for testing, there are "
                                f"{ahead} more operators preceding this one that are awaiting evaluation.\n"
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            with _VERIFY_SCHEDULER.acquire(req.devices, _notify_wait) as scheduled_devices:
                proc, json_path, assigned_devices = _run_verify_isolated(
                    op=req.op,
                    reference_path=reference_path,
                    kernel_code_path=kernel_code_path,
                    language=req.language,
                    devices=scheduled_devices,
                    stream=True,
                )
                q.put(
                    json.dumps(
                        {
                            "type": "log",
                            "data": (
                                "Current task is using device(s): "
                                f"{', '.join(assigned_devices) if assigned_devices else 'default device'}\n"
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                # 先把基本信息发给客户端
                q.put(
                    json.dumps(
                        {
                            "type": "meta",
                            "op": req.op,
                            "reference_path": reference_path,
                            "kernel_code_path": kernel_code_path,
                            "devices": assigned_devices,
                            "language": req.language,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                # Stream subprocess stdout as log events
                assert proc.stdout is not None
                for line in proc.stdout:
                    if not line:
                        continue
                    q.put(
                        json.dumps(
                            {
                                "type": "log",
                                "data": line,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                exit_code = int(proc.wait())

                # Load JSON result (best-effort)
                result: Any = None
                try:
                    if os.path.exists(json_path):
                        with open(json_path, "r", encoding="utf-8") as f:
                            result = json.load(f)
                except Exception as e:  # noqa: BLE001
                    result = [
                        {
                            "compiled": False,
                            "correctness": False,
                            "correctness_info": f"Failed to read verify output JSON: {e!r}",
                        }
                    ]
                    exit_code = 1
                finally:
                    try:
                        os.remove(json_path)
                    except OSError:
                        pass

                summary = summarize_result(result)

                q.put(
                    json.dumps(
                        {
                            "type": "result",
                            "exit_code": exit_code,
                            "summary": summary,
                            "result": result,
                            "reference_path": reference_path,
                            "kernel_code_path": kernel_code_path,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as e:  # noqa: BLE001
            q.put(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"{e!r}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        finally:
            q.put(sentinel)

    def event_stream() -> Iterable[str]:
        while True:
            item = q.get()
            if item is sentinel:
                break
            # item 是 JSON 字符串
            yield item  # type: ignore[misc]

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/verify_batch", response_model=List[VerifyBatchResponseItem])
def verify_batch_endpoint(req: VerifyBatchRequest) -> List[VerifyBatchResponseItem]:
    """
    批量并行版本的 AscendC kernel 评测接口。

    设计目标：
      - 一次请求中可提交多个 AscendC kernel（可能是不同 op / 不同代码路径）
      - 服务器在内部通过线程池并发调用 `_run_verify`，每个条目内部仍然使用
        multi-kernel-bench 的 Env 实现「并行编译 + 进程隔离测试」
      - 返回每个条目的独立结果，调用方可自行聚合或展示
    """
    _ensure_skill_root_env()

    responses: List[VerifyBatchResponseItem] = []

    # 为了在结果中保留原始顺序，这里记录索引
    with ThreadPoolExecutor(max_workers=min(8, len(req.items) or 1)) as executor:
        future_to_index: dict[Any, int] = {}
        for idx, item in enumerate(req.items):
            reference_path = os.path.abspath(item.reference_path)
            kernel_code_path = os.path.abspath(item.kernel_code_path)

            fut = executor.submit(
                _run_verify_isolated_scheduled,
                op=item.op,
                reference_path=reference_path,
                kernel_code_path=kernel_code_path,
                language=item.language,
                devices=item.devices,
            )
            future_to_index[fut] = idx

        # 预分配结果列表
        tmp_results: List[Optional[VerifyBatchResponseItem]] = [None] * len(req.items)

        for fut in as_completed(future_to_index):
            idx = future_to_index[fut]
            item = req.items[idx]
            reference_path = os.path.abspath(item.reference_path)
            kernel_code_path = os.path.abspath(item.kernel_code_path)

            try:
                exit_code, result, _logs, _assigned_devices = fut.result()
            except Exception as e:  # noqa: BLE001
                # 单个条目失败时，填充失败信息但不影响其它条目
                exit_code = 1
                result = [
                    {
                        "compiled": False,
                        "correctness": False,
                        "correctness_info": f"verify_batch internal error: {e!r}",
                    }
                ]

            summary = summarize_result(result)
            tmp_results[idx] = VerifyBatchResponseItem(
                op=item.op,
                reference_path=reference_path,
                kernel_code_path=kernel_code_path,
                exit_code=exit_code,
                result=result,
                summary=summary,
            )

    # 类型守护：理论上不会有 None，但为了安全过滤一下
    responses = [r for r in tmp_results if r is not None]
    return responses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Start AscendC kernel verify HTTP service (path-based)."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the HTTP service.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23457,
        help="Port to run the HTTP service on.",
    )
    parser.add_argument(
        "--devices",
        "--device-ids",
        dest="devices",
        type=str,
        default=None,
        help=(
            "Comma-separated NPU device ids or names, e.g. '0,1,2' or "
            "'npu:0,npu:1,npu:2'. Tasks will be assigned across this pool."
        ),
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=0,
        help=(
            "Maximum number of concurrent verify tasks. Defaults to the number "
            "of configured devices when --devices is provided."
        ),
    )
    args = parser.parse_args()

    if args.devices:
        os.environ["ASCEND_VERIFY_DEVICE_POOL"] = args.devices
    if args.max_inflight > 0:
        os.environ["ASCEND_VERIFY_MAX_INFLIGHT"] = str(args.max_inflight)

    _configure_scheduler(
        _parse_device_list(args.devices) if args.devices else _DEVICE_POOL,
        args.max_inflight,
    )

    print(
        f"Starting Ascend Kernel Verify Service on http://{args.host}:{args.port}/verify"
    )
    uvicorn.run(app, host=args.host, port=args.port)
