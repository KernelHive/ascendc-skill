### AIV 核上的 ND2NZ 格式转换

## 案例介绍

本案例展示了在矩阵乘算子场景中，使用 Matmul 高阶 API 进行计算，对内轴（内轴即矩阵的行方向）非 256 字节对齐的输入矩阵，在 AIV 核上进行 ND2NZ 格式转换对算子性能提升的效果。

为提升 Cube 单元的计算效率，ND 格式的输入矩阵在执行 Cube 计算前会先转换为 NZ 格式。ND 格式和 NZ 格式的具体内容可参考《Ascend C 算子开发指南》中的“算子实现 > 矩阵编程（高阶API） > 基础知识 > 数据格式”。

Matmul API 内部使用 DataCopy 随路格式转换同时进行格式转换以及数据搬运。但在数据非 256 字节对齐时，随路 ND2NZ 指令存在带宽利用率低的问题。因此输入矩阵的内轴非 256 字节对齐时，在进行 Matmul 计算前，利用 AIV 核上 Vector 计算单元完成 ND 格式到 NZ 格式的转换，可以避免随路非对齐数据搬运存在的效率低的问题，从而提升算子性能。

## AIV 核上的 ND2NZ 格式转换的适用场景

- 输入矩阵内轴非 256 字节对齐，且数据量较大影响随路格式转换的效率。

本案例的算子规格如下：

| 输入 | Shape       | Data type | Format |
|------|-------------|-----------|--------|
| a    | 1024, 1024  | float16   | ND     |
| b    | 1024, 4095  | float16   | ND     |

当前案例使用的 AI 处理器共 24 个核，算子中使能高阶 API Matmul 的纯 Cube 模式。使用 MDL 模板，Tiling 参数如下：

- 原始 shape：M=1024, N=4095, K=1024
- 单核 shape：singleCoreM=128，singleCoreN=1408，singleCoreK=1024
- 基本块 shape：baseM=128，baseN=256，baseK=64
- L1 缓存相关 Tiling 参数：stepM=1，stepN=1，stepKa=4，stepKb=4

## 获取性能数据

使用 msProf 工具获取算子仿真流水图和上板 Profiling 数据，重点分析 MTE2 的流水。

## 分析主要瓶颈点

- 优化前的 Cube 流水图如下，由于使用了随路 ND2NZ 指令，在 MTE2 数据搬运过程中进行数据格式的转换，导致 MTE2 整体占比较高。
- 优化前的 Profiling 数据如下，可以看到只使用 Cube 单元执行计算，aic_time 最大耗时 149.04us，其中 aic_mte2_ratio 占比很高。

## 设计优化方案

对于 ND 格式的输入矩阵，不再使用随路 ND2NZ 指令进行格式转换，而是利用 Vector 计算单元的能力完成数据格式转换。首先使用 DataCopyPad 接口，将非对齐的矩阵数据搬入 Unified Buffer，使用 Duplicate 接口填充需要补为对齐位置的数据，再逐行调用 Copy 接口实现数据从 ND 到 NZ 格式的重排，将重排后的 NZ 数据写入 workspace 内存，最后直接读取 workspace 上的 NZ 数据，进行 Matmul 计算。

AIV 核上的 ND2NZ 格式转换的完整样例请参考 Matmul 输入矩阵 ND 到 NZ 格式转换的算子样例。实现 AIV 核上的 ND2NZ 格式转换的主要步骤如下：

### 步骤 1

创建 Matmul 对象时，定义内轴非 256 字节对齐的 B 矩阵的 Format 为 NZ 格式。

```cpp
using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, ATYPE, true>;
// 使用 CubeFormat::NZ 定义矩阵 B 的类型信息
using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::NZ, BType, true>;
using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
using BIAS_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> matmulObj;
```

### 步骤 2

利用 Vector 计算单元实现 ND2NZ 格式转换。如下代码中 MatrixBtoNZ 为将 B 矩阵的 ND 格式转换为 NZ 格式的函数，该函数的具体实现请参考完整样例代码。

```cpp
// Vector ND2NZ
if ASCEND_IS_AIV {
    pipe->InitBuffer(ubBuf, TOTAL_UB_SIZE);
    MatrixBtoNZ<typename B_TYPE::T>(tempGM, bGMNZ, tiling, isTransB, ubBuf, tiling.baseK, tiling.baseN); // ND2NZ 格式转换函数
    SyncAll();
    // CV SYNC
    NotifyEvent<PIPE_MTE3>(4);
    return;
}
if ASCEND_IS_AIC {
    WaitEvent(4); // 等待 Vector 完成 ND2NZ 格式转换
}
```

### 步骤 3

设置左矩阵 A、右矩阵 B、Bias，完成矩阵乘操作。

```cpp
matmulObj.SetTail(tailM, tailN, shapes.k);
matmulObj.SetTensorA(aGlobal, false);
matmulObj.SetTensorB(bGlobal, false);
if (shapes.isBias) {
    matmulObj.SetBias(biasGlobal);
}
matmulObj.IterateAll(cGlobal);
```

## 验证优化方案性能收益

- 优化后的 Vector 流水图如下，利用 Vector 计算单元的能力，完成 B 矩阵的数据格式转换。
- 优化后的 Cube 流水图如下，不使用随路 ND2NZ 指令对 B 矩阵进行格式转换后，MTE2 的占比明显下降。
- 优化后的 Profiling 数据如下，可以看到同时使用 Cube 单元和 Vector 单元，aic_time 最大耗时 90.95us，其中 aic_mte2_ratio 占比明显降低。

| 优化方法         | 总耗时(us) | AIC_MTE2 平均耗时(us) | AIV_MTE2 平均耗时(us) |
|------------------|------------|-----------------------|-----------------------|
| 随路 ND2NZ       | 149.82     | 130.77                | 0                     |
| Vector 侧 ND2NZ  | 93.76      | 22.85                 | 10.31                 |

从上表中执行时间的对比，可以看出：不使用随路 ND2NZ 指令后，总耗时大幅下降，端到端性能提升明显。

## 总结

对于矩阵乘计算中矩阵内轴非 256 字节对齐的场景，随路 ND2NZ 指令的带宽利用率低，影响算子性能，通过在 AIV 核上进行 ND2NZ 的数据重排，提升算子整体性能。值得注意的是，带宽利用率与数据量有关，如果矩阵数据总量太小，即使是在 AIV 核上进行的 ND2NZ 转换也无法明显提升有效带宽，反而会因为引入了多核同步，导致算子端到端的性能劣化。
