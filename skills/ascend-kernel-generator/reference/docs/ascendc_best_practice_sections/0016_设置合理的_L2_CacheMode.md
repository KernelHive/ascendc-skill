### 设置合理的 L2 CacheMode

【优先级】高

## 说明

该性能优化指导适用于如下产品型号：

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 描述

L2 Cache 常用于缓存频繁访问的数据，其物理位置如下图所示：

L2 Cache 的带宽相比 GM 的带宽有数倍的提升，因此当数据命中 L2 Cache 时，数据的搬运耗时会优化数倍。通常情况下，L2 Cache 命中率越高，算子的性能越好，在实际访问中需要通过设置合理的 L2 CacheMode 来保证重复读取的数据尽量缓存在 L2 Cache 上。

## L2 Cache 访问的原理及 CacheMode 介绍

数据通过 MTE2 搬运单元搬入时，L2 Cache 访问的典型流程如下：

数据通过 MTE3 或者 Fixpipe 搬运单元搬出时，L2 Cache 访问的典型流程如下：

从上面的流程可以看出，当数据访问总量超出 L2 Cache 容量时，AI Core 会对 L2 Cache 进行数据替换。由于 Cache 一致性的要求，替换过程中旧数据需要先写回 GM（此过程中会占用 GM 带宽），旧数据写回后，新的数据才能进入 L2 Cache。

开发者可以针对访问的数据设置其 CacheMode，对于只访问一次的 Global Memory 数据设置其访问状态为不进入 L2 Cache，这样可以更加高效的利用 L2 Cache 缓存需要重复读取的数据，避免一次性访问的数据替换有效数据。

## 设置 L2 CacheMode 的方法

Ascend C 基于 GlobalTensor 提供了 SetL2CacheHint 接口，用户可以根据需要指定 CacheMode。

考虑如下场景，构造两个 Tensor 的计算，x 的输入 Shape 为 (5120, 5120)，y 的输入 Shape 为 (5120, 15360)，z 的输出 Shape 为 (5120, 15360)，由于两个 Tensor 的 Shape 不相等，x 分别与 y 的 3 个数据块依次相加。该方案主要为了演示 CacheMode 的功能，示例代码中故意使用重复搬运 x 的实现方式，真实设计中并不需要采用这个方案。下文完整样例请参考设置合理 L2 CacheMode 样例。

| 实现方案 | 原始实现 | 优化实现 |
|----------|----------|----------|
| 实现方法 | 总数据量 700MB，其中：x：100MB；y：300MB；z：300MB。使用 40 个核参与计算，按列方向切分。x、y、z 对应 GlobalTensor 的 CacheMode 均设置为 CACHE_MODE_NORMAL，需要经过 L2 Cache，需要进入 L2 Cache 的总数据量为 700MB。 | 总数据量 700MB，其中：x：100MB；y：300MB；z：300MB。使用 40 个核参与计算，按列方向切分。x 对应的 GlobalTensor 的 CacheMode 设置为 CACHE_MODE_NORMAL；y 和 z 对应的 GlobalTensor 的 CacheMode 设置为 CACHE_MODE_DISABLE。只有需要频繁访问的 x，设置为需要经过 L2 Cache。需要进入 L2 Cache 的总数据量为 100MB。 |

### 示例代码

**原始实现：**

```cpp
xGm.SetGlobalBuffer((__gm__ float *)x + AscendC::GetBlockIdx() * TILE_N);
yGm.SetGlobalBuffer((__gm__ float *)y + AscendC::GetBlockIdx() * TILE_N);
zGm.SetGlobalBuffer((__gm__ float *)z + AscendC::GetBlockIdx() * TILE_N);
```

**优化实现：**

```cpp
xGm.SetGlobalBuffer((__gm__ float *)x + AscendC::GetBlockIdx() * TILE_N);
yGm.SetGlobalBuffer((__gm__ float *)y + AscendC::GetBlockIdx() * TILE_N);
zGm.SetGlobalBuffer((__gm__ float *)z + AscendC::GetBlockIdx() * TILE_N);
// disable the L2 cache mode of y and z
yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
zGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
```

## 说明

你可以通过执行如下命令行，通过 msprof 工具获取上述示例的性能数据并进行对比。

```bash
msprof op --launch-count=2 --output=./prof ./execute_add_op
```

重点关注 Memory.csv 中的 `aiv_gm_to_ub_bw(GB/s)` 和 `aiv_main_mem_write_bw(GB/s)` 写带宽的速率。
