# 昇腾 NPU 迁移工作流（WorkFlow）

本文件与 [SKILL.md](SKILL.md) 中的章节顺序对齐，便于单独打开预览或分享给团队。

- **详细步骤与命令模板**：见 [SKILL.md](SKILL.md)。
- **交付文档模板**：迁移结束后维护 [mig_docs/](mig_docs/README.md) 下的 `Mig_report.md`、`Mig_Readme.md`、`Compare.md`。

---

## Workflow 图（Mermaid）

```mermaid
flowchart TD
  A[1 收集关键信息<br/>硬件/驱动-CANN/模型与 shape<br/>精度与性能目标/校准与评估数据<br/>工程依赖与部署形态] --> B[2 成功标准与基线<br/>对齐精度与延迟/吞吐判定规则]
  B --> C[3 可编译性预判<br/>IO 契约/算子与后处理/动态 shape<br/>最小分阶段验证计划]
  C --> D[4 环境准备与验证<br/>npu-smi + atc --version + 运行时依赖]
  D --> DS[环境快照表<br/>soc_version 依据]
  DS --> E{模型格式?}
  E -->|MindSpore| F[导出 MindIR]
  E -->|ONNX| G[保留 ONNX 路径]
  E -->|其他| T[转换为 ONNX 或 MindIR 等<br/>说明风险与版本]
  F --> PRE[编译前检查<br/>输入输出名/shape/dtype<br/>ONNX 核对 opset]
  G --> PRE
  T --> PRE
  PRE --> H[atc 生成 OM<br/>FP16 或 INT8 + 输入策略 + soc_version]
  H --> HV[编译后最小验证<br/>加载 OM + 1～3 样本前向<br/>记录日志与产物路径]
  HV --> I{目标精度 INT8?}
  I -->|否| J[7 部署运行与性能评测<br/>warmup + p50/p95 + 吞吐 + 日志]
  I -->|是| K[6 校准集与预处理对齐]
  K --> L[INT8 校准/量化配置]
  L --> J
  J --> GD[Golden 样本<br/>基线 vs NPU 数值一致性]
  GD --> M[8 全量精度与性能结论<br/>对照 Compare 口径]
  M --> N{是否通过?}
  N -->|是| O[交付物<br/>OM/配置/完整 atc 命令/测量复现]
  O --> MD[mig_docs<br/>Mig_report + Mig_Readme + Compare]
  N -->|否| P[9 风险定位与回滚迭代<br/>环境→shape→precision/量化→后处理<br/>算子替换或回退]
  P --> D

  %% Troubleshooting 回流（与 SKILL Troubleshooting 一致）
  P -. 编译失败 / 加载失败 .-> D
  P -. 精度大幅下降 尤其 INT8 .-> K
  P -. 性能不达标 .-> J
```

---

## 节点与 SKILL 章节对照

| 图中节点 | SKILL.md 对应 |
|----------|----------------|
| A | §1 先收集关键信息 |
| B | §2 定义成功标准与基线 |
| C | §3 可编译性预判 |
| D / DS | §4 准备环境并验证 + 环境快照表 |
| PRE / H / HV | §5 导出/编译 OM + 编译前检查 + 编译后最小验证 |
| K / L | §6 INT8 校准与量化 |
| J | §7 性能评估 |
| GD / M | §8 精度对比（含 Golden）与综合结论 |
| P | §9 风险点与回滚策略 |
| MD | `mig_docs` 规范输出 |

---

## 版本说明

- 图中 **FP16/INT8** 以实际目标为准；若先验证非量化通路，可在 `H` 阶段优先 FP16 再进入 `I`。
- `atc` 参数随 CANN 版本变化，以当前安装文档与用户 `atc --version` 为准。
