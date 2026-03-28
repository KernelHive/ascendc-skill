###### GetSpecialBasicConfig

## 功能说明

用于配置 SpecialBasicBlock 模板的参数，获取自定义 SpecialBasicBlock 模板。当前为预留接口。

## 函数原型

```cpp
__aicore__ constexpr MatmulConfig GetSpecialBasicConfig(
    const uint32_t basicM,
    const uint32_t basicN,
    const uint32_t basicK,
    const uint32_t singleCoreM,
    const uint32_t singleCoreN,
    const uint32_t singleCoreK,
    const uint32_t stepM,
    const uint32_t stepN,
    const bool intrinsicsLimit = false,
    const bool batchLoop = false,
    const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1
)
```

## 参数说明

本接口的所有参数用于设置 MatmulConfig 结构体中的参数，其中互相对应的参数的功能作用相同。

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| basicM | 输入 | 用于设置参数 basicM。<br>与 TCubeTiling 结构体中的 baseM 参数含义相同，Matmul 计算时 base 块 M 轴长度，以元素为单位。 |
| basicN | 输入 | 用于设置参数 basicN。<br>与 TCubeTiling 结构体中的 baseN 参数含义相同，Matmul 计算时 base 块 N 轴长度，以元素为单位。 |
| basicK | 输入 | 用于设置参数 basicK。<br>与 TCubeTiling 结构体中的 baseK 参数含义相同，Matmul 计算时 base 块 K 轴长度，以元素为单位。 |
| singleCoreM | 输入 | 用于设置参数 singleCoreM。<br>单核内 M 轴 shape 大小，以元素为单位。 |
| singleCoreN | 输入 | 用于设置参数 singleCoreN。<br>单核内 N 轴 shape 大小，以元素为单位。 |
| singleCoreK | 输入 | 用于设置参数 singleCoreK。<br>单核内 K 轴 shape 大小，以元素为单位。 |
| stepM | 输入 | 用于设置参数 stepM。<br>左矩阵在 A1 中缓存的 bufferM 方向上 baseM 的倍数。 |
| stepN | 输入 | 用于设置参数 stepN。<br>右矩阵在 B1 中缓存的 bufferN 方向上 baseN 的倍数。 |
| intrinsicsLimit | 输入 | 用于设置参数 intrinsicsCheck。<br>当左矩阵或右矩阵在单核上内轴（即尾轴）大于等于 65535（元素个数）时，是否使能循环执行数据从 Global Memory 到 L1 Buffer 的搬入。例如，左矩阵 A[M, K]，单核上的内轴数据 singleCoreK 大于 65535，配置该参数为 true 后，API 内部通过循环执行数据的搬入。参数取值如下：<br>• false：当左矩阵或右矩阵在单核上内轴大于等于 65535 时，不使能循环执行数据的搬入（默认值）。<br>• true：当左矩阵或右矩阵在单核上内轴大于等于 65535 时，使能循环执行数据的搬入。 |
| batchLoop | 输入 | 用于设置参数 isNBatch。<br>是否多 Batch 输入多 Batch 输出。仅对 BatchMatmul 有效，使能该参数后，仅支持 Norm 模板，且需调用 IterateNBatch 实现多 Batch 输入多 Batch 输出。参数取值如下：<br>• false：不使能多 Batch（默认值）。<br>• true：使能多 Batch。 |
| bmmMode | 输入 | 用于设置参数 batchMode。该参数用于 BatchMatmul 场景，关于 BatchMatmul 的介绍请参考 6.3.3.13 Batch Matmul 基础功能。<br>BatchMatmul 场景中 Layout 类型为 NORMAL 时，设置 BatchMatmul 输入 A/B 矩阵的多 batch 数据总和与 L1 Buffer 的大小关系。参数取值如下：<br>• BatchMode::BATCH_LESS_THAN_L1：多 batch 数据总和 < L1 Buffer Size；<br>• BatchMode::BATCH_LARGE_THAN_L1：多 batch 数据总和 > L1 Buffer Size；<br>• BatchMode::SINGLE_LARGE_THAN_L1：单 batch 数据总和 > L1 Buffer Size。 |

## 返回值说明

MatmulConfig 结构体。

## 约束说明

无
