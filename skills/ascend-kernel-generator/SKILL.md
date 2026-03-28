---
name: ascend-kernel-generator
description: >
  Generate, verify, and continuously improve AscendC custom kernels for user-specified
  operators using multi-kernel-bench.
metadata:
  short-description: AscendC kernel generation, verification, and golden solution management.
  version: "0.1.0"
---

## AscendC Kernel Generator Skill

This skill builds on `multi-kernel-bench` to **generate, verify, and continuously improve AscendC custom kernels** for user-specified operators.

The skill implements the following high-level loop:

1. **用户给出算子需求**（语义描述 / Torch 伪代码 / 输入输出规格）。
2. **将需求转写为 Torch 实现 + 算子原型定义**，并总结：
   - 算子名与类别（如 `add` / `mse_loss` / `matmul_add` 等）  
   - 输入 / 输出张量的形状、dtype、layout、广播/归约语义  
   - 数学公式与边界条件  
   - 约束与注意事项（数值稳定性、对齐、tile 要求等）  
3. **检索相似算子实现**（golden solutions、开源仓、官方文档），在此基础上生成与 `examples/add.py` 同格式的 AscendC kernel 工程描述文件。
4. 通过统一的评测接口（HTTP 服务 `/verify`，或其命令行客户端 `scripts/verify.py`）进行 **编译 + 正确性 + 性能** 验证。
5. **测试通过**：**Agent 应主动调用** `scripts/save_golden_solution.py`，将实现固化到 `reference/golden_solutions/`。
6. **测试失败**：基于错误信息检索 `reference/docs/`、`reference/ref_repositories/` 等资料进行修复，并重复验证与固化流程。

**请不要在model_src，python_bind_src中实现计算逻辑或者直接调用torch库中的函数，如at::conv2d，而应该在kernel_src中的Compute()函数来实现计算逻辑，使用完全自研核函数来实现算子**

## 1. Repository Layout

Key directories for this skill:

- `examples/`
  - `add_torch_reference.py`：Torch 参考实现示例。
  - `add.py`：与 multi-kernel-bench 集成的 AscendC 自定义算子完整示例（包含 `project_json_src`、`host_tiling_src`、`kernel_src` 等）。
- `scripts/`
  - `multi-kernel-bench/`：Env、Backend 与 AscendC 编译/执行管线。
  - `save_golden_solution.py`：将验证通过的实现保存到 `reference/golden_solutions/` 并更新 `info.json`。
  - `verify.py`：用于调用 HTTP 评测服务 `/verify` / `/verify_stream`，并在流式模式下将验证轨迹写入 `traj.json`。
  - `get_error_code_num.py`：对 `traj.json` 中的每一条记录进行分类，统计报错代码数量与索引、正确代码数量与索引。
  - `get_code_diff.py`：读取 `traj.json`，将错误代码与后续或最近的正确代码进行两两配对，生成 `error_fix_pairs.json`，每条记录包含 `idx`、`code_pair_name`、`op`、`error_idx`、`correct_idx`、`error_code`、`code_diff`、`error`、`summary`。
  - `get_content.py`：给定 `error_fix_pairs.json` 路径和 `idx`，打印对应索引的完整内容，便于 Agent 阅读和总结。
  - `extract_error_fix_into_experience.py`：给定 `error_fix_pairs.json` 路径、`idx` 与一段简短文字，将该文字写入对应 pair 的 `summary` 字段。
  - `save_fix_traj.py`：从 `error_fix_pairs.json` 中抽取 `{op, error, code_diff, summary}`，写入 `skill_root/reference/issues/{op}.json`，形成可长期复用的错误修复经验库。
- `reference/`
  - `docs/`：Ascend C / CANN / ACL 等官方文档与最佳实践的分片文本。
    - `ascendc_dev_guide_sections/`
    - `ascendc_best_practice_sections/`
    - `acl_interface_ref_sections/`
  - `ref_repositories/`：开源仓作为算子模板与工程结构参考。
    - `ascend-samples/`
    - `cann-ops/`
    - `ops-math/`
    - `ops-nn/`
    - `ops-transformer/`
  - `golden_solutions/`：已通过验证的 AscendC 实现（如 `add.py`、`mse_loss.py` 等）与 `info.json`。
  - `issues/`：预留的错误与修复轨迹目录（由内部脚本维护，不直接暴露给用户）。
- `template.md`：单算子视角的 prompt 结构与工作流模板。

---

## 1.1 Skill 安装路径与内部记忆

本 Skill 可能以多种方式安装并被调用：

- 全局安装：`~/.claude/skills/ascend-kernel-generator/`
- 项目级安装：`<workspace>/.claude/skills/ascend-kernel-generator/`

**Skill 安装路径（记为 `skill_root`）用于存放内部记忆：**

- `skill_root/reference/golden_solutions/`：长期保存已通过验证的 AscendC kernel，实现能力增长。
- `skill_root/reference/issues/`：预留的错误与修复轨迹仓库（当前对上层接口隐藏）。

在调用本 Skill 的任何脚本前，上游系统应先定位 `skill_root`，并建议注入环境变量：

- `ASCEND_SKILL_ROOT=<skill_root>`

如果未设置该环境变量，`scripts/save_golden_solution.py` 与 `scripts/save_error_fix_traj.py` 会自动将 `skill_root` 解析为：

- `skill_root = parent_dir_of(this_scripts_folder)`

这样，即使 Skill 被安装到不同位置，内部记忆始终相对于安装路径进行读写。

---

## 2. Input: Operator Requirement

When the user invokes this skill, they should describe the desired operator. The model should first normalize this into a **structured spec**:

- **基本信息**
  - **op name**：短小且语义明确，如 `add`, `reduce_sum`, `leaky_relu`, `matmul_add`。
  - **category**：如 `"math"`, `"reduce"`, `"activation"`, `"matmul"`, `"loss"` 等。
  - **硬件信息**：系统使用的npu卡型号
- **输入 / 输出定义**
  - 输入张量列表：`name`, `dtype`, `shape`（是否可变）、`layout`（NCHW / ND / ...）、broadcast / reduce 规则。
  - 输出张量列表：`name`, `dtype`, `shape` 与输入的关系。
- **数学语义**
  - 用公式或伪代码精确描述算子：如 $y = x_1 + x_2$、$y = \\sum_i x_i$、$y = \\max(0, x) + \\alpha \\min(0, x)$ 等。
- **约束与注意事项**
  - 数值稳定性需求（如 `softmax`, `log_sum_exp`）。
  - 对齐、block size、tile 大小的硬件约束（如果已知）。
  - 支持的 dtype 列表（float16/float32/bfloat16 等）。

该结构化信息既用于 Torch 参考实现，也用于 AscendC kernel 设计与 `project_json_src` 描述。

---

## 3. Phase 1: Torch Reference Implementation

**目标**：将算子需求转写为 **可执行的 Torch 参考实现**。若用户给定一个torch的参考实现，则将该实现参考给定结构来进行修改；若用户已经给出符合结构（包含Model、get_inputs、get_init_inputs）的torch实现，则改名之后直接使用，同时后续严禁对该文件进行修改。

- 在**用户当前工作目录**或**用户指定的输出目录**下，为新算子创建参考文件，例如：
  - `./<op>_torch_reference.py`（或 `<user_output_dir>/<op>_torch_reference.py`）
- 参考 `examples/add_torch_reference.py` 的结构：
  - 定义 `Model(nn.Module)` 或 `ModelNew(nn.Module)`，在 `forward` 中实现算子逻辑。
  - 定义 `get_inputs()`：根据算子规格生成随机输入张量列表。
  - 如有需要，定义 `get_init_inputs()`：初始化所需的额外张量/参数。

**要求**：

- Torch 参考实现应严格对齐用户需求的数学语义与形状逻辑。
- 输入 / 输出 tensor 的顺序与名称要与后续 AscendC 算子保持一致。

---

## 4. Phase 2: Operator Prototype & AscendC Design

在完成 Torch 参考实现后，模型需要输出一个 **详细的算子设计说明**，包括：

- **算子原型定义**（对应 Ascend 自定义算子工程中的 JSON 描述）：
  - 对应 `examples/add.py` / `reference/golden_solutions/add.py` 中的 `project_json_src` 格式。
  - 为每个输入 / 输出指定：
    - `name`
    - `param_type`（`required` / `optional`）
    - `format`（如 `"ND"`, `"NCHW"` 等）
    - `type`（`float`, `half`, `int32` 等）
- **Host 侧实现设计**
  - `host_tiling_src`：tiling data 结构、BLOCK_DIM、TILE_NUM 等策略（可参考 `add` 与 `ascend-samples/operator/`）。
  - `host_operator_src`：InferShape / InferDataType 实现与 op 注册逻辑。
- **Kernel 侧实现设计**
  - `kernel_src` 中的类与核函数原型（如 `KernelAdd` + `add_custom`）。
  - 内部函数拆分：`Init` / `CopyIn` / `Compute` / `CopyOut`。
  - AscendC API 选型：如 `AscendC::GlobalTensor`, `AscendC::LocalTensor`, `AscendC::TPipe`, `AscendC::TQue`, `AscendC::Add` 等。
- **Python 绑定与前向模型**
  - `python_bind_src`：`TORCH_LIBRARY_IMPL` 与 `PYBIND11_MODULE` 的注册逻辑。
  - `model_src`：`ModelNew` 中调用 `custom_ops_lib.<op_name>` 的方式。

> **重要约束（禁止 hack 实现）：**  
> - 模型在生成Ascendc算子代码中的 `model_src`以及`python_bind_src`时，不得直接依赖 Torch 的高阶 API / 预构建算子（例如 `torch.nn.Linear`、`torch.matmul` 等）来「偷跑」算子逻辑；  
> - 需要将算子核心计算拆解为自定义 AscendC kernel，并通过 `custom_ops_lib` 中的函数在 `ModelNew` 里进行调用；  
> - 当需要初始化权重时，参考下面的方式在ModelNew中初始化；
```python
class ModelNew(torch.nn.Module):
    def __init__(self, ...):
        super(ModelNew, self).__init__()
        ...
        self.conv2d = torch.nn.Conv2d(...)
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.xxx(x, self.conv2d.weight,...)
```

在这一阶段，模型应使用 **bash / grep / ripgrep** 等工具检索相似算子以做参考：

- 在 golden solutions 中查找：
  - `reference/golden_solutions/*.py`
- 在开源仓中查找：
  - `reference/ref_repositories/ascend-samples/`
  - `reference/ref_repositories/cann-ops/`
  - `reference/ref_repositories/ops-math/`
  - `reference/ref_repositories/ops-nn/`
  - `reference/ref_repositories/ops-transformer/`

示例命令（由上游工具执行，在仓库根目录下）：

```bash
rg "AddCustom" reference -n
rg "reduce_sum" reference/ref_repositories -n
```

---

## 5. Phase 3: Generate AscendC Kernel Description File

**核心产物**：一个与 `examples/add.py` / `reference/golden_solutions/add.py` 格式一致的 Python 文件，包含以下全局变量：

- `project_json_src`
- `host_tiling_src`
- `host_operator_src`
- `kernel_src`
- `python_bind_src`
- `model_src`

推荐输出位置（应位于**用户当前目录或用户指定目录**）：

- `./<op>.ascendc.py` 或 `./tmp/<op>.py`

约束：

- 保持与 Torch 参考实现的 I/O 顺序与 dtype 对齐。
- 确保 `project_json_src` 中的 `op` 名称与自定义 op 工程内的类/函数名相匹配（如 `AddCustom` / `add_custom`）。
- 遵循 AscendC 最佳实践，合理设置：
  - BLOCK_DIM
  - tile 数量与 tile 长度
  - `DataCopy`、`Add`、`Mul` 等算子 API 的使用方式
  - **请不要在model_src，python_bind_src中实现计算逻辑或者直接调用torch库中的函数，如at::conv2d，而应该在kernel_src中的Compute()函数来实现计算逻辑，使用完全自研核函数来实现算子**。

---

## 6. Phase 4: Verification Pipeline

生成 Torch 参考实现和 AscendC kernel 描述文件后，需要对实现进行 **编译 + 正确性 + 性能** 验证。

在推荐的部署形态下，评估由一个独立的 **HTTP 评测服务** 驱动，该服务封装在 `scripts/multi-kernel-bench/verify_server.py` 中，
对外暴露统一的 `/verify` 接口，内部仍然调用 multi-kernel-bench 的 `Env` 管线。上游系统通常不直接操作 HTTP，而是通过命令行客户端 `scripts/verify.py` 进行封装调用。

**重要约束（禁止直接执行底层 multi-kernel-bench 校验脚本）：**

- 所有编译 / 正确性 / 性能验证 **必须通过统一的评估接口触发**，推荐形态是通过命令行客户端 `scripts/verify.py`，其内部再调用 HTTP 服务 `/verify`。

### 6.1 命令行客户端 `scripts/verify.py`（推荐）

为了方便在命令行或简单脚本中调用 HTTP 评测服务，本仓库提供了一个轻量客户端，上游推荐统一通过该脚本完成评估调用：

- 位置：`scripts/verify.py`
- 依赖：标准库 `urllib`，通过 HTTP 调用 `/verify`。
- 支持的参数：
  - `--op`：算子名称，例如 `add`、`mse_loss`；
  - `--reference_path`：Torch 参考实现文件路径，例如 `examples/add_torch_reference.py`；
  - `--kernel_code_path`：AscendC kernel 描述文件路径，例如 `examples/add.py`；
  - `--url`：可选，HTTP 评测服务地址，默认 `http://127.0.0.1:23457/verify`，可通过环境变量 `ASCEND_VERIFY_URL` 覆盖。若与 `--stream` 一起使用，且 URL 以 `/verify` 结尾，会自动切换到 `/verify_stream`。
  - `--stream`：可选，启用流式接口 `/verify_stream`，实时接收评测日志（NDJSON 协议）。若未显式传入 `--stream`，但 `--url` 或 `ASCEND_VERIFY_URL` 以 `/verify_stream` 结尾，客户端会自动启用流式模式。
  - `--result_json_path`：最终评测结果 JSON 的保存路径。

示例用法（在仓库根目录）：

```bash
python scripts/verify.py \
  --op add \
  --reference_path examples/add_torch_reference.py \
  --kernel_code_path examples/add.py \
  --stream \
  --result_json_path ./add_verify_result.json
```

该脚本仅作为 HTTP 服务的客户端封装，**不直接参与 multi-kernel-bench 的内部逻辑**，符合本节前述“所有验证通过统一评估接口触发”的约束。

---

## 7. Phase 5: Golden Solution Persistence

当通过 HTTP 服务 `/verify`（或命令行客户端 `scripts/verify.py`）返回成功（`exit_code=0` 且 `correctness=True`）时，**Agent 应主动调用 `scripts/save_golden_solution.py`** 将当前实现固化到 **Skill 安装目录下的内部记忆** 中。

**重要**：虽然 `scripts/verify.py` 在验证成功时会自动尝试保存 golden solution（通过内部调用 `save_golden_solution.py`），但为了更好的功能解耦和流程控制，**强烈推荐 Agent 在验证成功后显式调用 `save_golden_solution.py`**。

### 7.1 调用方式

```bash
# 由上游系统预先导出 ASCEND_SKILL_ROOT 指向本 Skill 的安装路径
export ASCEND_SKILL_ROOT=/path/to/ascend-kernel-generator

python "$ASCEND_SKILL_ROOT/scripts/save_golden_solution.py" \
  --op <op_name> \
  --result_path <验证结果JSON文件路径> \
  --kernel_code_path <算子的ascendc实现文件路径> \
  --reference_path <算子的torch实现文件路径>
```

### 7.2 保存内容

该脚本会（相对于 `skill_root` 操作）：

- 将 AscendC kernel 描述文件复制到 `skill_root/reference/golden_solutions/<op>.py`。
- 在 `skill_root/reference/golden_solutions/info.json` 中记录：
  - `performance`：性能指标
  - `hardware`：硬件信息
  - `saved_at`：保存时间戳
  - `ref_path`：相对路径（如果提供了 `--reference_path`，当前版本中此字段被注释）

golden solutions 将作为下一次算子生成与修复时的优先参考项。

---

## 8. Phase 6: Failure Analysis & Fix Loop

当验证失败（`exit_code!=0`）时，进入 **错误分析–修复循环**。

### 8.1 检索与修复

基于 `compile_info` / `correctness_info` 中的关键信息，使用 grep/rg/bash 检索：
- **golden solutions**：
  - `reference/golden_solutions/*.py`
- **开源仓与官方示例**：
  - `reference/ref_repositories/ascend-samples/`
  - `reference/ref_repositories/cann-ops/`
  - `reference/ref_repositories/ops-math/`
  - `reference/ref_repositories/ops-nn/`
  - `reference/ref_repositories/ops-transformer/`
- **官方文档分片**：
  - `reference/docs/ascendc_dev_guide_sections/`
  - `reference/docs/ascendc_best_practice_sections/`
  - `reference/docs/acl_interface_ref_sections/`

模型应输出 **三段式分析**：

1. 错误类型与根因猜测（编译错误 / 运行错误 / 精度错误 / 性能异常）。
2. 参考依据（指出从哪些文件或文档中找到的类似问题与修复方式）。
3. 具体修复方案（指出需要修改 AscendC kernel、tiling、host 逻辑或 Torch 参考逻辑中的哪一部分）。

然后在原始 AscendC 描述文件中进行 **最小必要修改**，避免完全重写。

### 8.2 错误→修复经验抽取与固化

在多次尝试修复同一算子的过程中，`scripts/verify.py` 的流式模式会将每次尝试的代码与结果追加到 `traj.json` 中（参见 `verify.py` 中对 `traj` 的写入逻辑）。基于该轨迹文件，本 Skill 提供了一套标准化的错误修复经验抽取流程：

1. **统计错误/正确代码分布**  
   使用：
   ```bash
   python scripts/get_error_code_num.py --traj_path /path/to/traj.json
   ```  
   - 输出错误代码数量与索引、正确代码数量与索引；  
   - 若错误代码数量或正确代码数量任意一方为 0，则后续经验抽取流程不再继续。

2. **构造错误-正确代码配对与差异**  
   使用：
   ```bash
   python scripts/get_code_diff.py --traj_path /path/to/traj.json --output ./error_fix_pairs.json
   ```  
   - 为每条错误代码寻找一个对应的正确代码（优先选择索引更大的后续成功记录），生成形如  
     `{"idx", "code_pair_name", "op", "error_idx", "correct_idx", "error_code", "code_diff", "error", "summary": ""}`  
     的 pair 结构，并写入 `error_fix_pairs.json`；  
   - 其中 `code_diff` 为统一 diff 文本，`error` 尝试从验证结果中的错误信息/原因字段中抽取，`summary` 初始为空字符串。

3. **按需查看单条错误-修复 pair 内容**  
   使用：
   ```bash
   python scripts/get_content.py --json_path ./error_fix_pairs.json --idx <k>
   ```  
   - 按索引 `idx` 打印 `error_fix_pairs.json` 中的某一条记录，方便 Agent 阅读 `error_code`、`code_diff` 与 `error` 等字段。

4. **撰写精简的错误修复总结 summary**  
   对于每一条值得保留的错误-修复 pair，Agent 应基于第 3 步的输出，总结一段**简短、可复用**的文字描述（错误原因+错误分析+修复方案）。  
   然后调用：
   ```bash
   python scripts/extract_error_fix_into_experience.py \
     --pairs_path ./error_fix_pairs.json \
     --idx <k> \
     --summary "<short_fix_summary>"
   ```  
   将该 summary 写入对应 pair 的 `summary` 字段中。

5. **固化为 issues 经验库条目**  
   当对当前 `error_fix_pairs.json` 中的若干条 pair 完成 summary 填写后，调用：
   ```bash
   export ASCEND_SKILL_ROOT=/path/to/ascend-kernel-generator
   python scripts/save_fix_traj.py \
     --op <op_name> \
     --pairs_path ./error_fix_pairs.json
   ```  
   该脚本会：
   - 解析 `ASCEND_SKILL_ROOT`（或自动推断安装根目录）；
   - 将每条 pair 压缩为 `{op, error, code_diff, summary}` 结构；
   - 写入 `skill_root/reference/issues/<op>.json`，作为后续算子生成/修复阶段可检索的经验。

通过上述步骤，可以在不污染用户工作目录的前提下，将多次失败→成功的修复过程沉淀为结构化经验，用于指导后续同类算子的自动修复与生成。

## 9. Phase 7: 后处理

最终会话结束前，无论生成成功与否，都需要清理 `skill_root/scripts/multi-kernel-bench/tmp` 目录。但用户路径下的内容要进行保留。

使用脚本一键清理：

```bash
python3 skill_root/scripts/cleanup_tmp.py
```