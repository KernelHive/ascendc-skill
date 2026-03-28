### Matmul 高阶 API 使能 MDL 模板

## 案例介绍

本案例呈现了在矩阵乘算子场景中，使用 Matmul 高阶 API 进行矩阵乘法计算，使能 MDL 模板对算子性能的提升效果。在 MDL 模板中，MTE2 流水从 Global Memory 到 A1/B1 的数据搬运为一次性大包搬运，即一次 MTE2 能搬入多个 Matmul 计算的基本块，提升带宽利用率，使后续的 MTE1 流水尽可能复用 A1/B1 内基本块的缓存数据，减少 MTE2 的搬运次数。MDL 模板的详细介绍请参考《Ascend C 算子开发指南》中的“API 参考 > Ascend C API > 高阶 API > Matmul > Matmul > MatmulConfig”。

- **MDL 模板的适用场景**  
  一般适用于 MTE2 循环搬运次数多的大 shape 场景，MDL 模板在 A1/B1 中缓存多次计算需要的数据，避免 MTE2 频繁搬运。

- **MDL 模板的约束条件**  
  MDL 模板的 TCubeTiling 结构体需要满足 TCubeTiling 约束条件和 MDL 模板补充约束条件，具体请参考“API 参考 > Ascend C API > 高阶 API > Matmul > Matmul > TCubeTiling 结构体”。

## 算子规格

| 输入 | Shape       | Data type | Format |
|------|-------------|-----------|--------|
| a    | 128, 1024   | float16   | ND     |
| b    | 1024, 30720 | float16   | ND     |

当前案例使用的 AI 处理器共 24 个核，每个核中包含 1 个 AIC 核和 2 个 AIV 核。

## Tiling 参数

- **原始 shape**：M=128, N=30720, K=1024
- **单核 shape**：按 24 个 AIC 核进行切分，singleCoreM=128，singleCoreN=1280，singleCoreK=1024  
  对于 B 矩阵，沿着 N 轴进行切分，切分成 24 份的 singleCoreN，单核上处理 K * SingleCoreN 大小的数据。对于 A 矩阵，M 轴不进行切分即 singleCoreM=M，单核上处理 singleCoreM * K 大小的数据。总共 24 个核参与计算。
- **基本块 shape**：baseM=128，baseN=256，baseK=64
- **L1 相关 Tiling 参数**：stepM=1，stepN=1，stepKa=4，stepKb=4，depthA1=8，depthB1=8

## 获取性能数据

使用 msProf 工具获取算子仿真流水图和上板 Profiling 数据，因为 MDL 模板主要优化 MTE2 搬运效率，重点分析 MTE2 的流水情况。

## 分析主要瓶颈点

- **优化前的 Profiling 数据**如下，Matmul 默认为 Norm 模板。从 C 列的 aic_time 数据可以看出，多个核中最大算子执行耗时为 83.68us。从 C 列的 aic_time、L 列的 aic_mte2_time 和 M 列的 aic_mte2_ratio 几组数据来看，MTE2 平均耗时 75.64us，耗时占比高达 92% 以上，因此需要优化 MTE2 流水的耗时。

- **优化前的流水图**如下，MTE2 分多次从 Global Memory 搬运基本块到 A1/B1。由于输入的矩阵 Shape 较大，MTE2 循环搬运的次数多，但每次只搬运 1 个基本块，导致带宽利用率低，整体的 MTE2 搬运耗时长。进而影响后续的 MTE1 和 MMAD 流水，导致流水之间同步等待时间偏长。如红框所示，第一个基本块（baseM*baseN）的计算需要调用 16 次 MMAD 指令（singleCoreK/baseK=16），从左侧的第 1 个 MMAD 指令调用开始，到右侧的第 16 个 MMAD 指令调用结束，期间耗时 10.899us，其中大部分是流水同步等待耗时。

## 设计优化方案

下图是默认的 Norm 模板的 Matmul 计算流水示意图，MTE2 分多次从 Global Memory 搬运基本块到 A1 或 B1，每次只搬运一个基本块。Norm 模板的优势为启动开销小，可以提前启动 MTE1 流水；Norm 模板的劣势为在大 Shape 场景，MTE2 搬运次数多，搬运带宽利用率低，整体性能开销大。

**图 6-27 默认 Norm 模板流水示意图**

实现 Norm 模板的具体步骤如下：

**步骤 1** 创建 Matmul 对象，使用默认的 Norm 模板参数 CFG_NORM。

```cpp
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"

using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>;
using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>;
using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
using BIAS_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM> matmulObj; // 使用 CFG_NORM 定义 Matmul 对象
```

下图是 MDL 模板的 Matmul 计算流水示意图，MTE2 一次性从 Global Memory 搬运多个基本块到 A1 或 B1，每次搬运 stepM * stepKa 个基本块到 A1 或搬运 stepN * stepKb 个基本块到 B1。MDL 模板的优势为 MTE2 一次性搬运多个基本块，带宽利用率高，后续的 MTE1 流水能尽可能复用 A1 或 B1 的缓存数据，MTE2 重复搬运次数少。MDL 模板的劣势为 MTE2 头开销时间较长，MTE1 流水需要等待 MTE2 流水完成后才启动，MTE1 启动时间晚。

**图 6-28 MDL 模板流水示意图**

Matmul API 使能 MDL 模板的完整样例请参考 Matmul API 性能优化样例。使能 MDL 模板的主要步骤如下：

**步骤 1** 创建 Matmul 对象，使用默认的 MDL 模板参数 CFG_MDL。

```cpp
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"

using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>;
using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>;
using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
using BIAS_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> matmulObj; // 使用 CFG_MDL 定义 Matmul 对象
```

## 验证优化方案性能收益

- **优化后的 Profiling 数据**如下，从 C 列的 aic_time 数据可以看出，多个核中最大算子执行耗时为 53.4us，相较于优化前的 83.68us 有较大提升。从 L 列的 aic_mte2_time 数据可以看出，MTE2 平均耗时下降较多，从优化前的 75.64us 降低至 46.24us。

- **优化后的流水图**如下，MDL 模板相较于默认的 Norm 模板，MTE2 可以一次性搬运多个基本块，整体的 MTE2 搬运次数减少了。同时因为 MTE2 一次搬运多个基本块到 A1/B1，后续的 MTE1 流水能尽量复用 A1/B1 的缓存数据，减少了流水同步等待，提升了算子整体性能。如红框所示，第一个基本块（baseM*baseN）的计算需要调用 16 次 MMAD 指令（singleCoreK/baseK=16），从左侧的第 1 个 MMAD 指令调用开始，到右侧的第 16 个 MMAD 指令调用结束耗时约 5.198us，较优化前的 10.899us 提升较大，其中流水同步等待时间大幅减少。

## 总结

大 Shape 输入、MTE2 搬运次数多，且 MTE1 流水等 MTE2 流水的同步等待耗时较长的场景下，可以使能 MDL 模板。实现 MTE2 从 Global Memory 一次性搬入多个基本块到 A1 或 B1，使后续的 MTE1 流水能尽量复用 A1/B1 的缓存数据，减少 MTE2 的搬运次数，提升算子性能。
