### Matmul 高阶 API 使能 MTE2 Preload

## 案例介绍

本案例展示了在矩阵乘算子场景中，使用 Matmul 高阶 API 进行矩阵乘法计算，并通过使能 MTE2 Preload 提升算子性能的效果。

通过 MatmulConfig 中的 `doMTE2Preload` 参数开启矩阵 M/N 方向的预加载功能。预加载即在 MTE2 间隙提前加载 A 矩阵/B 矩阵数据，开启后可减少 MTE2 间隙，从而提升算子性能。

`doMTE2Preload` 参数的详细介绍请参考《Ascend C 算子开发指南》中的“API 参考 > Ascend C API > 高阶API > Matmul > Matmul > MatmulConfig”。

### 使能 MTE2 Preload 的适用场景

- MTE2 流水间隙较大，且 M 或 N 数值较大时。

### 使能 MTE2 Preload 的约束条件

- 仅在使用 MDL 模板和 SpecialMDL 模板时，MTE2 Preload 有效。
- 开启 M 或 N 方向预加载功能时，需保证 K 方向数据全载，且 M 或 N 方向开启 DoubleBuffer。
- K 方向数据全载的条件是：`singleK <= baseK * stepK`。
- M 方向开启 DoubleBuffer 的条件是：`depthA1 = stepM * stepK * 2`。
- N 方向开启 DoubleBuffer 的条件是：`depthB1 = stepN * stepK * 2`。

## 算子规格

| 输入 | Shape     | Data Type | Format |
|------|-----------|-----------|--------|
| a    | 128, 512  | float16   | ND     |
| b    | 512, 24576| float16   | ND     |

当前案例使用的 AI 处理器共 24 个核，算子中使能高阶 API Matmul 的纯 Cube 模式，使用 MDL 模板。

Tiling 参数如下：

- **原始 shape**：M=128, N=24576, K=512
- **单核 shape**：singleCoreM=128，singleCoreN=1024，singleCoreK=512
- **基本块 shape**：baseM=128，baseN=128，baseK=64
- **L1 缓存相关 Tiling 参数**：stepM=1，stepN=1，stepKa=8，stepKb=8，depthA1=8，depthB1=16

## 获取性能数据

使用 msProf 工具获取算子仿真流水图和上板 Profiling 数据，重点分析 Cube、Fixpipe 的流水情况。

## 分析主要瓶颈点

- 优化前的流水图显示，M 和 K 方向全载，因此 A 矩阵只搬运一次。由于 N 较大，B 矩阵会搬运多次，单次 MTE2 间存在间隙。
- 优化前的 Profiling 数据显示，aic_time 平均耗时 30.88us。

## 设计优化方案

使能 MTE2 Preload 功能：在创建 Matmul 对象时，开启 `doMTE2Preload` 开关。

具体步骤如下：

**步骤 1** 配置 MDL 模板参数，将 `doMTE2Preload` 参数设置为 2，使能 N 方向 Preload 功能。

```cpp
// preloadMode = 2
static constexpr MatmulConfig MM_CFG = GetMDLConfig(false, false, preloadMode);
```

**步骤 2** 基于自定义 MatmulConfig 模板参数，创建 Matmul 对象。

```cpp
AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
                AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
                AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>,
                AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>, MM_CFG> matmulObj;
```

## 验证优化方案性能收益

- 优化后的流水图显示，Tiling 参数不变，下一次计算使用的 B 矩阵数据提前加载，MTE2 间的间隙缩短。
- 优化后的 Profiling 数据显示，aic_time 平均耗时 28.50us，较优化前的 30.88us 有所提升。

## 总结

当 MTE2 流水间隙较大，且 M 或 N 数值较大时，可以考虑使能 MTE2 Preload 功能，提前加载 A 矩阵或 B 矩阵数据。
