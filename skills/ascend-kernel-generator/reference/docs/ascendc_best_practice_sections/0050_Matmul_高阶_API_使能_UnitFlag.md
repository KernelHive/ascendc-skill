### Matmul 高阶 API 使能 UnitFlag

## 案例介绍

本案例呈现了在矩阵乘算子场景中，使用 Matmul 高阶 API 进行矩阵乘法计算，使能 UnitFlag 功能对算子性能的提升效果。UnitFlag 功能为 AIC 核中 MMAD 计算指令和 FIXPIPE 数据搬运指令提供了基于内存访问的细粒度同步，使计算与搬运流水并行。使能 UnitFlag 功能的方式为将 MatmulConfig 中的 `enUnitFlag` 参数设置为 `true`。

`enUnitFlag` 参数的详细介绍请参考《Ascend C 算子开发指南》中的“API参考 > Ascend C API > 高阶API > Matmul > Matmul > MatmulConfig”。

### 使能 UnitFlag 的适用场景

算子的 MMAD 流水和 FIXPIPE 流水之间串行执行，FIXPIPE 等待 MMAD 计算完成才搬出结果，这个指令同步等待的时间在算子整体执行耗时中占比较高。这种场景可以使能 UnitFlag 功能，以获得 MMAD 和 FIXPIPE 流水并行的性能收益。

如果算子原本的 MMAD、FIXPIPE 流水可以被其他流水掩盖（比如 MTE2 Bound），这时使能 UnitFlag 功能总体收益很小。

### 使能 UnitFlag 的约束条件

- UnitFlag 功能仅支持 Norm、IBShare、MDL 三个模板。
- 使能 UnitFlag 功能时，不支持算子内同时存在 CO1(L0C) 搬出到 Global Memory 和 A1(L1) 搬出到 Global Memory 的两种流水。
- 使能 UnitFlag 功能时，若同时使能 L0C 累加功能，不支持多次 Iterate 计算、一次 GetTensorC 输出。

## 算子规格

本案例的算子规格如下：

| 输入 | Shape       | Data type | Format |
|------|-------------|-----------|--------|
| a    | 128, 64     | float16   | ND     |
| b    | 64, 30720   | float16   | ND     |

当前案例使用的 AI 处理器共 20 个核，每个核包含 1 个 AIC 核和 2 个 AIV 核。

算子的 Tiling 参数如下：

- **原始 shape**：M=128, N=30720, K=64
- **单核 shape**：按 20 个 AIC 核进行切分，singleCoreM=128，singleCoreN=1536，singleCoreK=64

对于 B 矩阵，沿着 N 轴进行切分，切分成 20 份 singleCoreN，单核上处理 K * SingleCoreN 大小的数据。对于 A 矩阵，M 轴不进行切分即 singleCoreM=M，单核上处理 singleCoreM * K 大小的数据。总共 20 个核参与计算。

- **基本块 shape**：baseM=128，baseN=256，baseK=64
- **L1 相关 Tiling 参数**：stepM=1，stepN=1，stepKa=4，stepKb=4，depthA1=8，depthB1=8

## 获取性能数据

使用 msProf 工具获取算子仿真流水图和上板 Profiling 数据。因为 UnitFlag 功能主要优化 MMAD 和 FIXPIPE 流水串行问题，所以获取性能数据后重点分析 Cube、FIXPIPE 的流水情况。

## 分析主要瓶颈点

- 优化前的流水图如下。如下图中红框所示，每一轮 MMAD 计算流水和 FIXPIPE 数据搬出流水之间都是串行执行的，完成 MMAD 计算后才开始 FIXPIPE 数据搬出，考虑实现 MMAD 与 FIXPIPE 之间流水并行来优化算子性能。

- 优化前的 Profiling 数据如下，从 C 列的 aic_time 数据可以看出，多个核中最大算子执行耗时为 37.39us。

## 设计优化方案

如下图所示，未开启 UnitFlag 功能时，MMAD 和 FIXPIPE 是指令级别的同步，FIXPIPE 指令需要等 MMAD 指令执行完成才进行结果搬出，MMAD 和 FIXPIPE 之间流水串行。

**图 6-29 未开启 UnitFlag 功能**

如下图所示，开启 UnitFlag 功能时，MMAD 和 FIXPIPE 指令是 512B 大小的细粒度同步。在一条 MMAD 指令执行过程中，每当完成一个 512B 数据结果的计算，FIXPIPE 立即开始搬出该 512B 的数据，从而实现 MMAD 和 FIXPIPE 之间的流水并行，提升算子性能。

**图 6-30 开启 UnitFlag 功能**

Matmul API 使能 UnitFlag 功能的完整样例请参考 Matmul API 性能优化样例。使能 UnitFlag 功能的主要步骤如下：

### 步骤 1

自定义 MatmulConfig 模板参数，将其中的 `enUnitFlag` 参数设置为 `true`，使能 UnitFlag 功能。

```cpp
__aicore__ inline constexpr MatmulConfig GetCustomMDLCFG()
{
    auto mmCfg = CFG_MDL;
    mmCfg.enUnitFlag = true;
    return mmCfg;
}
constexpr static MatmulConfig CUSTOM_CFG_MDL = GetCustomMDLCFG();
```

### 步骤 2

基于自定义的 MatmulConfig 模板参数，创建 Matmul 对象。

```cpp
using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>;
using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>;
using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
using BIAS_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CUSTOM_CFG_MDL> matmulObj;
```

## 验证优化方案性能收益

- 优化后的流水图如下，MMAD 计算流水和 FIXPIPE 数据搬出流水之间实现了流水并行。

- 优化后的 Profiling 数据如下，从 C 列的 aic_time 数据可以看出，多个核中最大算子执行耗时为 34.66us，较优化前的 37.39us 有约 7.3% 的性能提升。

## 总结

在算子的 MMAD 计算流水和 FIXPIPE 数据搬出流水串行且未被其他流水掩盖（比如 MTE2 Bound）时，考虑使能 UnitFlag 功能，实现 MMAD 计算流水和 FIXPIPE 数据搬出流水的流水并行，提升算子性能。
