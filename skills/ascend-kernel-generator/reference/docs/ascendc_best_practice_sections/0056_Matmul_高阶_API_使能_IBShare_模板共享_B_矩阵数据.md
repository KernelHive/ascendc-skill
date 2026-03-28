### Matmul 高阶 API 使能 IBShare 模板共享 B 矩阵数据

## 案例介绍

本案例呈现了在矩阵乘算子场景中，使用 Matmul 高阶 API 进行矩阵乘法计算，B 矩阵使能 IBShare 对算子性能的提升效果。IBShare 功能通过共享 L1 Buffer 上相同的 A 矩阵或 B 矩阵数据，减少重复的 MTE2 数据搬运开销，提升算子性能。该功能支持 A 矩阵和 B 矩阵其中一个矩阵使能 IBShare，也支持 A 矩阵和 B 矩阵同时使能 IBShare。

## 使能 IBShare 的适用场景

MIX 场景（包含矩阵计算和矢量计算）下，多个 AIV 的 A 矩阵或 B 矩阵 GM 地址相同，且多个 AIV 复用的 A 矩阵或 B 矩阵在 L1 Buffer 上全载。

## 使能 IBShare 的约束条件

- A 矩阵和 B 矩阵同时使能 IBShare 的场景，同一算子中其它 Matmul 对象的 A 矩阵和 B 矩阵也必须同时使能 IBShare。
- A 矩阵和 B 矩阵同时使能 IBShare 的场景，获取矩阵计算结果时，只支持调用 IterateAll 接口，且只支持输出到 Global Memory。

## 算子规格

| 输入 | Shape     | Data type | Format |
|------|-----------|-----------|--------|
| a    | 64, 384   | float16   | ND     |
| b    | 384, 256  | float16   | ND     |

当前案例使用的 AI 处理器共 20 个核，每个核中包含 1 个 AIC 核和 2 个 AIV 核。因为输入 shape 较小，本案例以单核为示例，参考 SetDim 接口在 MIX 模式下的使用，在 Tiling 程序中设置参与运算的核数为 2。

Tiling 参数如下：

- **原始 shape**：M=64, N=256, K=384
- **单核 shape**：singleCoreM=32，singleCoreN=256，singleCoreK=384。A 矩阵拆成两半，一半在 AIV0 上处理，一半在 AIV1 上处理；AIV0 和 AIV1 使用的 B 矩阵数据相同
- **基本块 shape**：baseM=32，baseN=256，baseK=64
- **L1 缓存相关 Tiling 参数**：stepM=1，stepN=1，stepKa=6，stepKb=6

## 获取性能数据

使用 msProf 工具获取算子仿真流水图和上板 Profiling 数据。因为 IBShare 功能主要是通过共享 L1 Buffer 上相同的 A 矩阵或 B 矩阵数据，减少重复的 MTE2 数据搬运开销，所以重点分析 MTE2 的流水情况。

## 分析主要瓶颈点

- 优化前的流水图如下，不使能 IBShare 模板，默认使用的 Norm 模板。黑框标识 AIV0 发起的 MTE2 搬运流水：MTE2 总共搬运了 12 次，其中 A 矩阵搬运了 6 次（stepM*stepKa=6），B 矩阵搬运了 6 次（stepN*stepKb=6）。红框标识的 AIV1 发起的 MTE2 搬运流水，跟 AIV0 基本一致。在该案例中，因为 AIV1 使用的 B 矩阵跟 AIV0 使用的 B 矩阵数据相同，且 singleCoreN=baseN*stepN，singleCoreK=baseK*stepKb，即 B 矩阵可以在 L1 全载。考虑在 AIV0 搬入 B 矩阵到 L1 Buffer 后，将 B 矩阵数据缓存在 L1 Buffer 上等待 AIV1 进行复用，进而节省 B 矩阵的 MTE2 重复搬运开销。

- 优化前的 Profiling 数据如下，C 列的 aic_time 是 10.29us，K 列的 aic_mte2_time 是 5.56us。

## 设计优化方案

下图是不使能 IBShare 模板（默认使用 Norm 模板）的 Matmul 计算流水示意图。MTE2 分多次从 Global Memory 搬运基本块到 A1 或 B1，即使前后两次搬运的 B 矩阵基本块数据是相同的数据，也会重复搬运。

**图 6-39 不使能 IBShare 模板的 Matmul 流水示意图**

下图是使能 IBShare 模板的 Matmul 计算流水示意图。MTE2 分多次从 Global Memory 搬运基本块到 A1 或 B1，若前后两次搬运的 B 矩阵基本块数据相同，不会重复搬运，第一次搬运到 B1 内的数据会被复用。

**图 6-40 使能 IBShare 模板的 Matmul 流水示意图**

Matmul API 使能 IBShare 模板共享 B 矩阵的完整样例请参考仅 B 矩阵使能 IBShare 样例。

使能 IBShare 功能的主要步骤如下：

### 步骤 1：创建 Matmul 对象

```cpp
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"

using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>;
using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, false,
LayoutMode::NONE, true>; // 设置B矩阵的IBSHARE参数为true
using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
using BIAS_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_IBSHARE_NORM> matmulObj; // 使用默认的IBShare模板参数CFG_IBSHARE_NORM定义Matmul对象
```

## 验证优化方案性能收益

- 优化后的流水图如下，黑框标识的 AIV0 发起的 MTE2 搬运流水，与优化前一致。红框标识的 AIV1 发起的 MTE2 搬运流水，相较于优化前的 A 矩阵和 B 矩阵一共 12 次 MTE2 数据搬运，减少到了仅 6 次 A 矩阵的 MTE2 数据搬运，省去了 B 矩阵的 6 次 MTE2 数据搬运开销。

- 优化后的 Profiling 数据如下，C 列的 aic_time 是 9.93us，较优化前的 10.29us 提升了 3.55%。K 列的 aic_mte2_time 是 4.71us，较优化前的 5.56us 提升了 15.46%。

## 总结

MIX 场景（包含矩阵计算和矢量计算）下，若多个 AIV 的 A 矩阵或 B 矩阵 GM 地址相同，且多个 AIV 复用的 A 矩阵/B 矩阵在 L1 Buffer 上全载。可以考虑使能 IBShare 模板，通过共享 L1 Buffer 上相同的 A 矩阵或 B 矩阵数据，减少重复的 MTE2 数据搬运开销，提升算子性能。
