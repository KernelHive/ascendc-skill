# mig_docs 说明

本目录为**昇腾迁移交付文档模板**。执行迁移时建议：

1. **复制**整个 `mig_docs/` 到目标仓库根目录（或 `docs/mig_docs/`），与代码一并版本管理。
2. 按模板填写三份文档（文件名保持统一，便于检索）：

| 文件 | 用途 |
|------|------|
| [Mig_report.md](Mig_report.md) | 迁移过程与**变更清单**、环境快照、产物与 ATC 命令 |
| [Mig_Readme.md](Mig_Readme.md) | 迁移后**训练（若适用）与推理**复现说明 |
| [Compare.md](Compare.md) | **精度与性能**对比（基线 vs 昇腾，含测量口径） |

> 文件名说明：迁移报告为 **`Mig_report.md`**（report，非 reprot）。
