## AscendC Kernel Verify HTTP Service

本仓库除了 MCP Server 以外，还提供了一个 **HTTP 版本的 AscendC kernel 评测服务**，
方便通过普通 HTTP 客户端或自建 Agent 编排来调用 multi-kernel-bench 的 Env 管线。

服务实现位于：`scripts/multi-kernel-bench/verify_server.py`。

---

### 1. 依赖准备

- **Python 依赖**

在仓库根目录执行：

```bash
cd /home/huangzixiao/.claude/skills/ascend-kernel-generator

# multi-kernel-bench 运行依赖
pip install -r scripts/multi-kernel-bench/requirements.txt

# HTTP 服务依赖
pip install fastapi uvicorn pydantic
```

- **Ascend/CANN 环境**

确保本机已正确安装并配置：

- CANN / AscendC 编译环境
- NPU 驱动与运行时

要求与直接运行 `scripts/multi-kernel-bench/verify.py`、`scripts/multi-kernel-bench` 相同。

---

### 2. 启动 HTTP 评测服务

在仓库根目录（或任意位置）执行：

```bash
export ASCEND_SKILL_ROOT=/home/huangzixiao/.claude/skills/ascend-kernel-generator

cd /home/huangzixiao/.claude/skills/ascend-kernel-generator
python scripts/multi-kernel-bench/verify_server.py --host 0.0.0.0 --port 23457
```

启动成功后，会监听：

- `POST http://<host>:<port>/verify`

日志中会打印类似：

```text
Starting Ascend Kernel Verify Service on http://0.0.0.0:23457/verify
```

---

### 3. HTTP 接口约定

- **URL**

```text
POST /verify
Content-Type: application/json
```

- **请求体（JSON）**

```json
{
  "op": "add",
  "reference_path": 算子Torch参考实现文件路径,
  "kernel_code_path": 算子Ascendc实现文件路径,
  "language": "ascendc",
  "devices": ["npu:0"]
}
```

字段说明：

- `op` (`string`, 必填)：算子名称，例如 `"add"`, `"mse_loss"`。
- `reference_path` (`string`, 必填)：Torch 参考实现文件路径，通常是 `<op>_torch_reference.py` 的绝对路径或相对路径。
- `kernel_code_path` (`string`, 必填)：AscendC kernel 描述文件路径，格式参考 `examples/add.py`。
- `language` (`string`, 选填，默认 `"ascendc"`): 后端语言。
- `devices` (`array[string]`, 选填)：设备列表，例如 `["npu:0"]`；缺省时由 Env 自行选择。

- **返回体（JSON）**

```json
{
  "exit_code": 0,
  "reference_path": 算子Torch参考实现文件路径,
  "kernel_code_path": 算子Ascendc实现文件路径,
  "result": [
    {
      "compiled": true,
      "correctness": true,
      "performance": {
        "latency_ms": 0.12
      },
      "hardware": {
        "device": "npu:0"
      }
    }
  ],
  "summary": "compiled=True, correctness=True, hardware=npu:0, latency=0.12ms"
}
```

字段说明：

- `exit_code` (`number`)：`0` 表示编译 + 正确性通过，非 0 表示失败。
- `reference_path` (`string`)：在磁盘中写入的 Torch 参考实现绝对路径。
- `kernel_code_path` (`string`)：写入的 AscendC 描述文件绝对路径。
- `result` (`array[object]`)：multi-kernel-bench `Env.step` 的原始结果列表。
- `summary` (`string`)：从 `result` 派生的人类可读摘要。

其语义与本 Skill 中 Phase 4 所描述的验证流程完全一致，仅调用通道从本地脚本调用换成了 HTTP/JSON。

python scripts/verify.py \
  --op add \
  --reference_path examples/add_torch_reference.py \
  --kernel_code_path examples/add.py \
  --devices npu:0 \
  --stream