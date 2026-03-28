#### Batch Matmul 基础功能

## 功能介绍

Batch Matmul 是指批量处理 Matmul 计算的场景。该场景对外提供了 IterateBatch 调用接口，调用一次 IterateBatch，可以计算出多个 singleCoreM * singleCoreN 大小的 C 矩阵。

Matmul 单次计算的过程需要搬入和搬出数据，当进行多次 Matmul 计算且单次 Matmul 计算的输入 shape 较小时，搬运开销在整体耗时中占比较大。通过 IterateBatch 接口批量处理 Matmul，可以有效提升带宽利用率。

Batch Matmul 当前支持 4 种 Layout 类型：

- BSNGD
- SBNGD
- BNGS1S2
- NORMAL（BMNK 的数据排布格式）

相关数据排布格式请参考 IterateBatch。

下图为 NORMAL 数据排布格式的 Batch Matmul 计算。整个 Matmul 计算一共包含 4 个矩阵乘操作：

- mat_a1 * mat_b1
- mat_a2 * mat_b2
- mat_a3 * mat_b3
- mat_a4 * mat_b4

需要单核上计算四个 singleCoreM * singleCoreN。在该场景下，如果 shape 较小，可以将其视为 Batch Matmul 场景进行批量处理，以提升性能。一次 IterateBatch 可同时计算出：

- mat_c1 = mat_a1 * mat_b1
- mat_c2 = mat_a2 * mat_b2
- mat_c3 = mat_a3 * mat_b3
- mat_c4 = mat_a4 * mat_b4

图 6-41 NORMAL 数据排布格式的 Batch Matmul 示意图

## 使用场景

Matmul 计算需要计算出多个 singleCoreM * singleCoreN 大小的 C 矩阵，且单次 Matmul 计算处理的 shape 较小。

## 约束说明

- 只支持 Norm 模板。
- 对于 BSNGD、SBNGD、BNGS1S2 Layout 格式，输入 A、B 矩阵按分形对齐后的多 Batch 数据总和应小于 L1 Buffer 的大小；对于 NORMAL Layout 格式没有这种限制，但需通过 MatmulConfig 配置 batchMode 参数，即输入 A、B 矩阵多 Batch 数据大小与 L1 Buffer 的大小关系。
- 对于 BSNGD、SBNGD、BNGS1S2 Layout 格式，称左矩阵、右矩阵的 G 轴分别为 ALayoutInfoG、BLayoutInfoG，则 ALayoutInfoG / batchA = BLayoutInfoG / batchB；对于 NORMAL Layout 格式，batchA、batchB 必须满足倍数关系。Bias 的 shape(batch, n) 中的 batch 必须与 C 矩阵的 batch 相等。
- 如果接口输出到 Unified Buffer 上，输出 C 矩阵大小 BaseM * BaseN 应小于分配的 Unified Buffer 内存大小。
- 对于 BSNGD、SBNGD Layout 格式，输入输出只支持 ND 格式数据。对于 BNGS1S2、NORMAL Layout 格式，输入支持 ND/NZ 格式数据。
- Batch Matmul 不支持量化/反量化模式，即不支持 SetQuantScalar、SetQuantVector 接口。
- BSNGD 场景，不支持一次计算多行 SD，需要算子程序中循环计算。
- 异步模式不支持 IterateBatch 搬运到 Unified Buffer 上。
- 模板参数 enableMixDualMaster（默认取值为 false）设置为 true，即使能 MixDualMaster（双主模式）场景时，不支持 Batch Matmul。
- 在 batch 场景，A 矩阵、B 矩阵支持 half/float/bfloat16_t/int8_t 数据类型，不支持 int4b_t 数据类型。

## 调用示例

以下是 NORMAL 数据排布格式的 Batch Matmul 调用示例。BSNDG 数据排布格式的 Batch Matmul 完整示例请参考 BatchMatmul 样例。

### Tiling 实现

使用 SetBatchInfoForNormal 设置 A/B/C 的 M/N/K 轴信息和 A/B 矩阵的 BatchNum。

```cpp
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform);
int32_t M = 32;
int32_t N = 256;
int32_t K = 64;
tiling->SetDim(1);
tiling->SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
matmul_tiling::DataType::DT_FLOAT16);
tiling->SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
matmul_tiling::DataType::DT_FLOAT16);
tiling->SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
matmul_tiling::DataType::DT_FLOAT);
tiling->SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
matmul_tiling::DataType::DT_FLOAT);
tiling->SetShape(M, N, K);
tiling->SetOrgShape(M, N, K);
tiling->EnableBias(true);
tiling->SetBufferSpace(-1, -1, -1);

constexpr int32_t BATCH_NUM = 3;
tiling->SetBatchInfoForNormal(BATCH_NUM, BATCH_NUM, M, N, K); // 设置矩阵排布
tiling->SetBufferSpace(-1, -1, -1);

optiling::TCubeTiling tilingData;
int ret = tiling.GetTiling(tilingData);
```

### Kernel 实现

#### 创建 Matmul 对象

通过 MatmulType 设置输入输出的 Layout 格式为 NORMAL。

```cpp
#include "lib/matmul_intf.h"

typedef AscendC::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, half, false,
LayoutMode::NORMAL> aType;
typedef AscendC::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, half, true,
LayoutMode::NORMAL> bType;
typedef AscendC::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, float, false,
LayoutMode::NORMAL> cType;
typedef AscendC::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
constexpr MatmulConfig MM_CFG = GetNormalConfig(false, false, false,
BatchMode::BATCH_LESS_THAN_L1);
AscendC::Matmul<aType, bType, cType, biasType, MM_CFG> mm;
```

#### 初始化操作

```cpp
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling); // 初始化matmul对象
```

#### 设置左矩阵 A、右矩阵 B、Bias

```cpp
mm.SetTensorA(gm_a); // 设置左矩阵A
mm.SetTensorB(gm_b); // 设置右矩阵B
mm.SetBias(gm_bias); // 设置Bias
```

#### 完成矩阵乘操作

左矩阵每次计算 batchA 个 MK 数据，右矩阵每次计算 batchB 个 KN 数据。

```cpp
mm.IterateBatch(gm_c, batchA, batchB, false);
```

#### 结束矩阵乘操作

```cpp
mm.End();
```
