### Matmul 高阶 API 使能 Tiling 全量常量化

## 案例介绍

本案例呈现了在使用 Matmul 高阶 API 进行矩阵乘法计算时，使能 Matmul Tiling 全量常量化对算子性能的提升效果。Matmul API 在初始化和迭代过程中有大量 Scalar 计算，Matmul 初始化时的 Scalar 计算影响指令头开销，Matmul 迭代间的 Scalar 计算可能阻塞 MTE2 流水。在调用 Matmul API 实现矩阵乘法时，使用 `MatmulApiStaticTiling` 参数替代 `TCubeTiling` 变量参数，将 Scalar 计算提前到编译期进行，以减少运行时的 Scalar 计算开销，实现算子性能的提升。关于 `MatmulApiStaticTiling` 的内容，请参考《Ascend C 算子开发指南》中的“API 参考 > Ascend C API > 高阶 API > Matmul > Matmul > Matmul 模板参数”。

### Matmul Tiling 常量化的适用场景

- Matmul 初始化时的 Scalar 计算较多，影响指令头开销。
- Matmul 迭代之间的 Scalar 计算较多，阻塞 MTE2 流水。

### Matmul Tiling 常量化场景

Matmul Tiling 常量化需要在编译期确定部分 Tiling 参数，根据确定参数的不同，分为全量常量化和部分常量化两种场景。使用 Matmul Tiling 常量化需要满足两种场景中任一场景的条件：

- **全量常量化**：能够确定常量 singleCore Shape（singleCoreM/singleCoreN/singleCoreK）和常量 base Shape（basicM/basicN/basicK，也称 baseM/baseN/baseK）。
- **部分常量化**：能够确定常量 base Shape（basicM/basicN/basicK，也称 baseM/baseN/baseK）。

其中，全量常量化场景比部分常量化场景可以减少更多的 Scalar 计算开销。

## 算子规格

| 输入 | Shape      | Data Type | Format |
|------|------------|-----------|--------|
| a    | 128, 64    | float16   | ND     |
| b    | 64, 30720  | float16   | ND     |

当前案例使用的 AI 处理器共 24 个核，每个核中包含 1 个 AIC 核和 2 个 AIV 核。

## Tiling 参数

- **原始 shape**：M=128, N=30720, K=64。
- **单核 shape**：按 24 个 AIC 核进行切分，singleCoreM=128，singleCoreN=1280，singleCoreK=64。
  - 对于 B 矩阵，沿着 N 轴进行切分，切分成 24 份的 singleCoreN，单核上处理 K * singleCoreN 大小的数据。
  - 对于 A 矩阵，M 轴不进行切分即 singleCoreM=M，单核上处理 singleCoreM * K 大小的数据。
  - 总共 24 个核参与计算。
- **基本块 shape**：baseM=128，baseN=256，baseK=64。
- **L1 相关 Tiling 参数**：stepM=1，stepN=1，stepKa=4，stepKb=4，depthA1=8，depthB1=8。

## 获取性能数据

使用 msProf 工具获取算子仿真流水图和上板 Profiling 数据。相较于基础场景，Tiling 常量化在编译期期间将部分或全部 Tiling 参数由变量转化为常数值，在算子执行时直接使用常量化的 Tiling 参数，可以减少 Scalar 性能开销，所以重点分析 Scalar 流水。

## 分析主要瓶颈点

- **优化前的流水图**：默认不使能 Tiling 常量化，Tiling 参数需要从 Host 侧拷贝到 Kernel 侧，导致 Matmul 初始化时的 Scalar 计算较多，第一个 MTE2 指令开始于 3.536us 左右，MTE2 前的指令头开销在算子整个流水中占比偏大，因此需要优化 Scalar 计算。
- **优化前的 Profiling 数据**：
  - 从 C 列的 `aic_time` 数据来看，多个核中最大算子执行耗时为 10.62us。
  - 从 G 列的 `aic_scalar_time` 数据来看，Scalar 平均耗时 6.32us。

## 设计优化方案

### 默认不使能 Tiling 常量化的流程

默认不使能 Tiling 常量化功能时，开发者在 host 侧创建 Tiling 对象，通过调用 API 自动获取 Tiling 参数。然后将 Tiling 参数从 Host 侧传递到 Kernel 侧，在 Kernel 侧初始化操作时传入。在算子执行时，使用 Tiling 变量参数完成矩阵乘操作。

### 使能 Tiling 常量化的流程

使能 Tiling 常量化功能时，开发者只需要在 Kernel 侧创建 Matmul 对象时，调用 `GetMatmulApiTiling` 接口在编译期获取常量化 Tiling 信息，即可完成 Tiling 常量化。在算子执行时，使用常量化的 Tiling 参数完成矩阵乘操作，减少 Scalar 计算开销。

### 使能 Tiling 全量常量化的步骤

Matmul API 使能 Tiling 全量常量化的完整样例请参考 Matmul Tiling 常量化的算子样例。使能 Tiling 全量常量化功能的步骤如下：

#### 步骤 1：获取自定义 MatmulConfig 模板

调用获取 MatmulConfig 模板的接口 `GetMMConfig` 时，使用常数值设置 `MatmulShapeParams`，得到带有常量化参数的自定义 MatmulConfig 模板 `CUSTOM_CFG`。

```cpp
constexpr int32_t MAX_M = 10000; // custom matmul kernel support max value of M Dim shape
constexpr int32_t MAX_N = 10000; // custom matmul kernel support max value of N Dim shape
constexpr int32_t MAX_K = 10000; // custom matmul kernel support max value of K Dim shape
constexpr int32_t BASE_M = 128;  // BASE_M * BASE_K * sizeof(typeA) <= L0A size
constexpr int32_t BASE_N = 256;  // BASE_N * BASE_K * sizeof(typeB) <= L0B size
constexpr int32_t BASE_K = 64;   // BASE_M * BASE_N * sizeof(typeC) <= L0C size

constexpr MatmulShapeParams shapeParams = {
    MAX_M,
    MAX_N,
    MAX_K,
    BASE_M,
    BASE_N,
    BASE_K
};

constexpr MatmulConfig CUSTOM_CFG = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(shapeParams);
```

#### 步骤 2：创建 Matmul 对象

首先调用 `GetMatmulApiTiling` 接口，将 Tiling 信息常量化，得到常量化模板参数 `CONSTANT_CFG`，包括常量化的 Matmul Tiling 信息和 MatmulConfig 模板。创建 Matmul 对象时，使用常量化模板参数 `CONSTANT_CFG`。

```cpp
using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>;
using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>;
using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>;
using BIAS_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>;

constexpr static auto CONSTANT_CFG = AscendC::GetMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(CUSTOM_CFG);
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CONSTANT_CFG> matmulObj;
```

#### 步骤 3：初始化操作

- **全量常量化场景**：可以在 `REGIST_MATMUL_OBJ` 接口的入参传递 Tiling 参数的位置，使用空指针替代。
- **部分常量化场景**：在 Kernel 侧使用 `REGIST_MATMUL_OBJ` 接口初始化 Matmul 对象时，仍需要使用 Tiling。

```cpp
// 全量常量化场景，初始化操作示例
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj, (TCubeTiling*)nullptr);

// 部分常量化场景，初始化操作示例
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj, &tiling);
```

## 验证优化方案性能收益

- **优化后的流水图**：通过使能 Tiling 全量常量化，无需将 Tiling 参数从 Host 侧拷贝到 Kernel 侧，在编译期完成 Tiling 常量化，减少了 Matmul 初始化时的 Scalar 计算。从 0us 起到第一个 MTE2 指令发起，这之间的时间为 Matmul 初始化时间，Matmul 初始化时间从优化前的 3.536us 减少到 2.185us，有一定提升。
- **优化后的 Profiling 数据**：
  - 从 C 列的 `aic_time` 数据来看，多个核中最大算子执行耗时为 7.87us，相较于优化前的 10.62us 提升了 25.9%。
  - 从 G 列的 `aic_scalar_time` 数据来看，Scalar 平均耗时 3.38us，相较于优化前的 6.32us 提升了 46.5%。

## 总结

算子在调用 Matmul API 完成矩阵乘计算时，若 Matmul 初始化时的 Scalar 计算较多，影响了指令头开销，或 Matmul 迭代间的 Scalar 计算较多，阻塞了 MTE2 流水。在这两类场景下，满足上文提及的 Tiling 常量化使能条件（全量常量化或部分常量化），可以考虑使能 Tiling 常量化，减少 Scalar 计算开销，提升算子性能。
