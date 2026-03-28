###### IterateNBatch

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | × |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

调用一次 `IterateNBatch`，会进行 N 次 `IterateBatch` 计算，计算出 N 个多 Batch 的 `singleCoreM * singleCoreN` 大小的 C 矩阵。在调用该接口前，需将 `MatmulConfig` 中的 `isNBatch` 参数设为 `true`，使能多 Batch 输入多 Batch 输出功能，并调用 `SetWorkspace` 接口申请临时空间，用于缓存计算结果，即 `IterateNBatch` 的结果输出至 `SetWorkspace` 指定的 Global Memory 内存中。

对于 BSNGD、SBNGD、BNGS1S2 的 Layout 格式，调用该接口之前需要在 tiling 中使用 `SetALayout`/`SetBLayout`/`SetCLayout`/`SetBatchNum` 设置 A/B/C 的 Layout 轴信息和最大 BatchNum 数；对于 Normal 数据格式则需使用 `SetBatchInfoForNormal` 设置 A/B/C 的 M/N/K 轴信息和 A/B 矩阵的 BatchNum 数。实例化 Matmul 时，通过 `MatmulType` 设置 Layout 类型，当前支持 3 种 Layout 类型：BSNGD、SBNGD、BNGS1S2。

## 函数原型

```cpp
template <bool sync = true, bool waitIterateBatch = false>
__aicore__ inline void IterateNBatch(
    const uint32_t batchLoop,
    uint32_t batchA,
    uint32_t batchB,
    bool enSequentialWrite,
    const uint32_t matrixStrideA = 0,
    const uint32_t matrixStrideB = 0,
    const uint32_t matrixStrideC = 0
)
```

## 参数说明

### 模板参数说明

| 参数名 | 描述 |
|--------|------|
| sync | 获取 C 矩阵过程分为同步和异步两种模式：<br>• 同步：需要同步等待 `IterateNBatch` 执行结束，后续由开发者自行获取输出到 Global Memory 上的计算结果。<br>• 异步：不需要同步等待 `IterateNBatch` 执行结束。<br>通过该参数设置同步或者异步模式：同步模式设置为 `true`；异步模式设置为 `false`。默认为同步模式。 |
| waitIterateBatch | 是否需要通过 `WaitIterateBatch` 接口等待 `IterateNBatch` 执行结束，仅在异步场景下使用。默认为 `false`。<br>`true`：需要通过 `WaitIterateBatch` 接口等待 `IterateNBatch` 执行结束，然后由开发者自行获取输出到 Global Memory 上的计算结果。<br>`false`：不需要通过 `WaitIterateBatch` 接口等待 `IterateNBatch` 执行结束。调用本接口后，需要调用 `GetBatchTensorC` 接口获取 C 矩阵，或者由开发者自行处理等待 `IterateNBatch` 执行结束的过程。 |

### 函数参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| batchLoop | 输入 | 当前计算的 BMM 个数。 |
| batchA | 输入 | 当前单次 BMM 调用计算左矩阵的 batch 数。 |
| batchB | 输入 | 当前单次 BMM 调用计算右矩阵的 batch 数，brc 场景 batchA/B 不相同。 |
| enSequentialWrite | 输入 | 输出是否连续存放数据。 |
| matrixStrideA | 输入 | A 矩阵源操作数相邻 nd 矩阵起始地址间的偏移，默认值是 0。 |
| matrixStrideB | 输入 | B 矩阵源操作数相邻 nd 矩阵起始地址间的偏移，默认值是 0。 |
| matrixStrideC | 输入 | 该参数预留，开发者无需关注。 |

## 返回值说明

无

## 约束说明

- 单 BMM 内计算遵循之前的约束条件。
- 对于 BSNGD、SBNGD、BNGS1S2 Layout 格式，输入 A、B 矩阵多 Batch 数据总和应小于 L1 Buffer 的大小。
- 当使能 MixDualMaster（双主模式）场景时，即模板参数 `enableMixDualMaster` 设置为 `true`，不支持使用该接口。

## 调用示例

实例功能：完成 aGM、bGM 矩阵乘，结果保存到 cGm 上，其中 aGM 数据的 layout 格式为 BSNGD，bGM 数据的 layout 格式为 BSNGD，cGM 的 layout 格式为 BNGS1S2，左矩阵每次计算 batchA 个 SD 数据，右矩阵每次计算 batchB 个 SD 数据。

```cpp
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

extern "C" __global__ __aicore__ void kernel_matmul_rpc_batch(
    GM_ADDR aGM,
    GM_ADDR bGM,
    GM_ADDR cGM,
    GM_ADDR biasGM,
    GM_ADDR tilingGM,
    GM_ADDR workspaceGM,
    uint32_t isTransposeAIn,
    uint32_t isTransposeBIn,
    int32_t batchA,
    int32_t batchB
) {
    // 定义 matmul type
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::BSNGD> aType;
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true, LayoutMode::BSNGD> bType;
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, false, LayoutMode::BNGS1S2> cType;
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
    SetAtomicNone();

    // 初始化 tiling 数据
    TCubeTiling tiling;
    auto tempTilingGM = (__gm__ uint32_t*)tilingGM;
    auto tempTiling = (uint32_t*)&tiling;
    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(int32_t); ++i, ++tempTilingGM, ++tempTiling) {
        *tempTiling = *tempTilingGM;
    }

    // 初始化 gm 数据
    AscendC::GlobalTensor<half> aGlobal;
    AscendC::GlobalTensor<half> bGlobal;
    AscendC::GlobalTensor<float> cGlobal;
    AscendC::GlobalTensor<float> biasGlobal;
    int32_t sizeA = tiling.ALayoutInfoB * tiling.ALayoutInfoS * tiling.ALayoutInfoN * tiling.ALayoutInfoG * tiling.ALayoutInfoD * sizeof(half);
    int32_t sizeB = tiling.BLayoutInfoB * tiling.BLayoutInfoS * tiling.BLayoutInfoN * tiling.BLayoutInfoG * tiling.BLayoutInfoD * sizeof(half);
    int32_t sizebias = tiling.CLayoutInfoB * tiling.CLayoutInfoN * tiling.CLayoutInfoG * tiling.CLayoutInfoS2 * sizeof(float);
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(aGM), sizeA);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(bGM), sizeB);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(biasGM), sizebias);
    tiling.shareMode = 0;
    tiling.shareL1Size = 512 * 1024;
    tiling.shareL0CSize = 128 * 1024;
    tiling.shareUbSize = 0;
    int offset_a = 0, offset_b = 0, offset_c = 0, offset_bias = 0;
    AscendC::GlobalTensor<A_T> gm_a;
    gm_a.SetGlobalBuffer(const_cast<__gm__ half*>(aGlobal[offset_a].GetPhyAddr()), tiling.ALayoutInfoS * tiling.ALayoutInfoN * tiling.ALayoutInfoG * tiling.ALayoutInfoD);
    AscendC::GlobalTensor<B_T> gm_b;
    gm_b.SetGlobalBuffer(const_cast<__gm__ half*>(bGlobal[offset_b].GetPhyAddr()), tiling.BLayoutInfoS * tiling.BLayoutInfoN * tiling.BLayoutInfoG * tiling.BLayoutInfoD);
    AscendC::GlobalTensor<BiasT> gm_bias;
    gm_bias.SetGlobalBuffer(const_cast<__gm__ float*>(biasGlobal[offset_bias].GetPhyAddr()), tiling.CLayoutInfoN * tiling.CLayoutInfoG * tiling.CLayoutInfoS2);

    // 创建 Matmul 实例
    AscendC::Matmul<aType, bType, cType, biasType> mm1;
    AscendC::TPipe pipe;
    g_cubeTPipePtr = &pipe;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1);
    mm1.Init(&tiling);
    int g_lay = tiling.ALayoutInfoG > tiling.BLayoutInfoG ? tiling.ALayoutInfoG : tiling.BLayoutInfoG;
    int for_extent = tiling.ALayoutInfoB * tiling.ALayoutInfoN * g_lay / tiling.BatchNum;
    mm1.SetTensorA(gm_a[0], isTransposeAIn);
    mm1.SetTensorB(gm_b[0], isTransposeBIn);
    mm1.SetWorkspace(workspaceGM, 0);
    if (tiling.isBias) {
        mm1.SetBias(gm_bias[0]);
    }

    // 多 batch Matmul 计算
    mm1.IterateNBatch(for_extent, batchA, batchB, false);
}
```
