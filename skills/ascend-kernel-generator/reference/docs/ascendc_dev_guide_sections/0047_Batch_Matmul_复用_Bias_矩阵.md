#### Batch Matmul 复用 Bias 矩阵

## 功能介绍

在 Batch Matmul 场景中，Matmul API 可以一次性计算出多个大小为 `singleCoreM * singleCoreN` 的 C 矩阵。当 Batch Matmul 场景有 Bias 输入时，默认的 Bias 输入矩阵包含 Batch 轴，即 Bias 的大小为 `Batch * N`。通过开启 Bias 复用功能，当每个 Batch 计算使用的 Bias 数据相同时，只需输入一个不带 Batch 轴的 Bias 矩阵。Batch Matmul 的 Bias 矩阵复用功能默认不启用，用户需要设置 `MatmulConfig` 中的 `isBiasBatch` 参数为 `false` 来开启此功能。

![图 6-42 带有 Batch 轴的 Bias 计算示意图]()

如上图所示，Batch Matmul 中未复用 Bias 矩阵的场景，每计算出一个 `singleCoreM * singleCoreN` 大小的 C 矩阵，都会与 `1 * singleCoreN` 大小的 Bias 矩阵相加。若不同 Batch 的计算使用的 Bias 数据相同，则多 Batch 计算可以复用同一个 Bias 矩阵，如下图所示，此场景中调用 `SetBias` 接口时，只需设置一个 `1 * singleCoreN` 大小的 Bias 矩阵。

![图 6-43 复用 Bias 计算示意图]()

## 使用场景

Batch Matmul 中每个 Batch 的 Matmul 计算可以使用相同的 Bias 矩阵。

## 约束说明

A、B、C 矩阵的 Layout 类型都为 NORMAL 时，不支持 `batchMode` 参数设为 `SINGLE_LARGE_THAN_L1`，即 Bias 复用场景下，单 Batch 的 A、B 矩阵数据总和不得超过 L1 Buffer 的大小。

## 调用示例

完整的算子样例请参考 BatchMatmul 复用 Bias 算子样例。

```cpp
// 自定义 MatmulConfig 参数，将其中的 isBiasBatch 参数设置为 false，使能 BatchMatmul 的 Bias 复用功能。
constexpr MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;
constexpr MatmulBatchParams batchParams = {
    false, BatchMode::BATCH_LESS_THAN_L1, false /* isBiasBatch */
};
constexpr MatmulConfig CFG_MM = GetMMConfig<configMode>(batchParams);
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MM> mm;

REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling); // 初始化 matmul 对象

mm.SetTensorA(gm_a); // 设置左矩阵 A
mm.SetTensorB(gm_b); // 设置右矩阵 B
mm.SetBias(gm_bias); // 设置 Bias，矩阵大小为 1 * singleCoreN
mm.IterateBatch(gm_c, batchA, batchB, false);
mm.End();
```
