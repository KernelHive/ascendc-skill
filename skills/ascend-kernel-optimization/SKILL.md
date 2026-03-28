---
name: ascendKernelOptimization
description: |
  Ascend C Kernel Code Optimization Skill.
  使用本地RAG知识库检索优化经验，对算子kernel代码进行迭代优化。
  核心流程: 识别bottleneck -> 检索RAG -> 生成修改 -> 评估 -> 迭代
  Trigger examples: "optimize kernel", "优化 kernel 代码".
---

# Ascend C Kernel Code Optimization

## Language Behavior / 语言行为

- 中文提问用中文回答，英文提问用英文回答
- 代码使用英文标识符

## Scope

- 优化 Ascend C 算子的 kernel 代码 (非 tiling 参数)
- 优化分析阶段必须同时查看该算子的 `op_host` 与 `op_kernel` 目录下全部代码文件
- 基于 RAG 知识库检索优化经验
- 迭代优化直至收敛或达到最大迭代次数
- 优化完成后保存结果到新目录: `/root/wjh/codex/output_ops/<run_id>/<category>/<op_name>`

## Workflow

### 1. 确认输入
- 算子类别和名称 (如 `math/abs`)
- tiling资源路径
- RAG知识库路径 (默认: `scripts/rag_db/optimization_points.json`)
- 算子源码可访问，且可完整读取 `op_host` 与 `op_kernel` 全部文件

### 2. 迭代优化流程 (核心)

每轮迭代执行以下步骤：

```
┌─────────────────────────────────────────────────────────────┐
│  1. 识别 Bottleneck                                        │
│     - 必须先完整读取 op_host 与 op_kernel 全部代码          │
│     - 联合分析 host-kernel 接口、数据流与调度关系           │
│     - 调用 LLM 分析当前代码                                 │
│     - 识别最关键的性能瓶颈                                  │
│     - 避免重复识别历史瓶颈                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. 检索 RAG 知识库                                        │
│     - 基于 bottleneck 描述检索                              │
│     - 从 optimization_points.json 获取 1 条最相关优化经验    │
│     - 包含: title, description, bottleneck, code_diff      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. 生成代码修改                                           │
│     - 结合 bottleneck 和 RAG 经验                          │
│     - 使用 LLM 生成 diff 格式修改                          │
│     - 解析 diff 应用到代码                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4. 评估                                                   │
│     - 调用评估服务获取性能分数                               │
│     - 分数越低越好                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  5. 更新最佳解                                             │
│     - 记录分数改善的修改                                    │
│     - 继续下一轮迭代                                        │
└─────────────────────────────────────────────────────────────┘
```

完成全部迭代后必须执行最终落盘：
- 创建新的 `run_id` 目录（格式建议 `run_YYYYMMDD_HHMMSS`，若冲突追加后缀）
- 将最佳结果保存到 `/root/wjh/codex/output_ops/<run_id>/<category>/<op_name>`
- 不覆盖历史输出目录

### 3. Bottleneck 识别指引

LLM 分析时关注以下类型瓶颈：

| 瓶颈类型 | 描述 |
|---------|------|
| 内存访问 | 非连续内存访问、bank冲突、未利用UB |
| 计算效率 | 未使用向量化、重复计算、冗余操作 |
| 分支预测 | 过多条件分支、代码布局不当 |
| 同步开销 | 不必要的全局同步、流水线停顿 |
| 数据搬运 | 主机-设备传输冗余、重复拷贝 |

### 4. RAG 检索

知识库文件：`scripts/rag_db/optimization_points.json`

每条经验包含：
- **title**: 优化技巧标题
- **description**: 详细描述
- **bottleneck**: 对应瓶颈描述
- **code_diff**: 参考修改片段

检索方式：
- 不依赖 embedding API
- 基于 bottleneck 文本做本地关键词/相似度匹配，返回最匹配的 1 条经验

### 5. Diff 格式

```diff
<<<< op_kernel/abs_kernel.cpp
// 原始代码
int32_t a = 1;
====
int32_t a = 8;  // 修改后
>>>>
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TILING_RESOURCE_BASE` | tiling资源基础路径 |
| `KNOWLEDGE_BASE_PATH` | RAG知识库路径 (默认: `scripts/rag_db` 或 `scripts/rag_db/optimization_points.json`) |
| `EVALUATE_SERVICE_URL` | 评估服务 (默认 http://127.0.0.1:6666/evaluate) |
| `LLM_BASE_URL` | LLM API 地址 |
| `LLM_API_KEY` | LLM API 密钥 |
| `LLM_MODEL` | LLM 模型名 (默认 DeepSeek-V3.2) |
| `MAX_ITERATIONS` | 最大迭代次数 (默认3) |
| `OUTPUT_BASE` | 输出根目录 (默认 /root/wjh/codex/output_ops，实际输出为 `<OUTPUT_BASE>/<run_id>/<category>/<op_name>`) |

## Usage

```python
from scripts.optimizer import create_optimizer

optimizer = create_optimizer(
    op_name="abs",
    op_category="math",
    tiling_resource_base="/path/to/tiling/resources",
)

best_code, best_score, output_dir = optimizer.optimize()
```

## Files

- `SKILL.md` - 技能定义
- `scripts/rag_db/` - RAG知识库数据
- `scripts/optimizer.py` - 主优化器
- `scripts/sampler.py` - 采样器 (bottleneck识别 + RAG检索 + 代码生成)
- `scripts/evaluator.py` - 评估器
- `scripts/knowledge_base.py` - RAG知识库
- `scripts/config.py` - 配置
