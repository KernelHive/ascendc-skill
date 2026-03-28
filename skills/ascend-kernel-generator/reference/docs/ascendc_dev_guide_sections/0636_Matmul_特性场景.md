###### Matmul 特性场景

除了前述介绍的 Matmul 基本计算能力外，还请掌握 Matmul 的基础知识和算子实现。  
另外，Matmul 矩阵编程还提供了适用于不同场景的处理能力及多种功能，具体场景和使用的关键接口或参数列于下表中，详细内容请见对应章节的介绍。

### 表 15-595 Matmul 特性表

| 特性分类 | 特性描述 | 涉及的关键 API 或参数 |
|----------|----------|----------------------|
| 功能实现 | 多核对齐场景 | `SetDim`、`EnableMultiCoreSplitK`（多核切 K 场景） |
| 功能实现 | 多核非对齐场景 | `SetTail`、`SetDim`、`EnableMultiCoreSplitK`（多核切 K 场景） |
| 功能实现 | 异步场景 | `Iterate`、`GetTensorC`、`IterateAll` |
| 功能实现 | CallBack 回调功能 | `MatmulCallBackFunc`、`SetUserDefInfo`、`SetSelfDefineData` |
| 功能实现 | 量化场景 | `SetDequantType`、`SetQuantScalar`、`SetQuantVector` |
| 功能实现 | ChannelSplit 功能 | MatmulConfig 模板参数中的 `isEnableChannelSplit` 参数 |
| 功能实现 | GEMV 场景 | `SetAType` |
| 功能实现 | Sparse Matmul 场景 | `SetSparse`、`SetSparseIndex` |
| 功能实现 | 上三角/下三角计算功能 | MatmulPolicy 模板参数 |
| 功能实现 | TSCM 输入场景 | `DataCopy` |
| 功能实现 | ND_ALIGN 输出功能 | `SetCType` |
| 功能实现 | Partial Out 功能 | MatmulConfig 模板参数中的 `isPartialOutput` 参数 |
| 功能实现 | 双主模式功能 | MatmulConfig 模板参数中的 `enableMixDualMaster` 参数 |

### 表 15-596 BatchMatmul 特性表

| 特性分类 | 特性描述 | 主要涉及的 API 接口 |
|----------|----------|----------------------|
| 功能实现 | BatchMatmul 基础场景 | NORMAL 排布格式的 BatchMatmul：`IterateBatch`、`SetBatchInfoForNormal` |
| 功能实现 | BSNGD、SBNGD、BNGS1S2 排布格式的 BatchMatmul | `IterateBatch`、`SetALayout`、`SetBLayout`、`SetCLayout`、`SetBatchNum` |
