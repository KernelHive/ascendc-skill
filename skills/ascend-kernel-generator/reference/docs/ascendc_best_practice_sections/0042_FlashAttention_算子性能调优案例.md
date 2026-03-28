## FlashAttention 算子性能调优案例

## 案例介绍

本案例中的算子 FlashAttentionScoreGrad，用于训练场景下计算注意力的反向输出，即 FlashAttentionScore 算子的反向计算。

已知注意力的正向计算公式为：

为方便表达，以变量 S 和 P 表示计算公式：

则注意力的反向计算公式为：

计算流程图如下：

图 6-1 算子计算流程

按照 FlashAttention 反向计算流程的实现，简介整体计算流程如下。对本算子的算法感兴趣的用户可简单了解，无需重点关注。

1. **重计算 p**：本步骤重计算了 FlashAttention 流程中的 softmax 结果 p，计算结果保存在 ub 中。
2. **计算 dp**：该计算包含 matmul 计算和 dropout 计算，matmul 计算中，左矩阵为 dy，右矩阵为转置后的 value。
3. **计算 ds**：本计算中，FlashSoftmaxGrad 计算的入参为 dy、正向输出 attention_in，该结果与 dp 做减操作，最终的结果与 p 相乘得到结果 ds。
4. **计算 dq**：本计算将 ds 结果与 key 做 matmul 计算，并将结果与 scale 相乘得到结果 dq。
5. **计算 dk**：本计算将转置后的 ds 结果与 query 做 matmul 计算，并将结果与 scale 相乘得到结果 dk。
6. **计算 dv**：本计算将 p 的结果做 drop 计算，转置后与 dy 做 matmul 计算。

本案例的验证平台为 Atlas A2 训练系列产品/Atlas A2 推理系列产品，以两个场景为例：

- **场景一**：输入维度信息为 B=1，N1=12，N2=12，S1=6144，S2=6144，D=128，causal 场景，即 atten_mask 的形状为下三角，如图 6-2。
- **场景二**：输入维度信息为 B=24，N1=5，N2=5，S1=9216，S2=9216，D=64，不带 atten_mask 和 drop_mask 输入。

主要涉及的优化手段包括 tiling 基本块大小调整、核间负载均衡、CV 流水并行、MTE2 流水优化以及 FixPipe 流水优化等。

图 6-2 causal 场景 atten_mask 形状

## 获取性能数据

流水优化分析工具包括 CAModel 和 Profiling 工具，分别从两个方面分析：

1. 从 Profiling 工具生成的 Profiling 数据中分析各项流水的占比。
2. 从 CAModel 工具生成的打点图分析各流水并行情况。

## 分析主要瓶颈点

通过观察分析流水图和 Profiling 数据，结合优化经验来判断性能瓶颈点。在优化过程中不同阶段可能会出现不同的瓶颈点，需要不断优化以达到最佳性能。

- 根据优化经验，循环间会存在一些不必要的性能开销，循环越多性能可能越差；满足 UB 最大空间限制的情况下，UB 切分的基本块越大，循环越少。算子中通过 InitBuffer 接口分配 UB buffer 大小。

```cpp
pipe->InitBuffer(ubBuffer, 120 * 1024);
pipe->InitBuffer(tmpBuffer, 30 * 1024);
pipe->InitBuffer(vecClc3, 8 * 1024);
```

如上代码所示，InitBuffer 接口的第二个参数表示 buffer 占用的大小，所有 buffer 大小的和即为占用的总空间。这里 `120 * 1024 + 30 * 1024 + 8 * 1024 = 158KB < UB Size`，没有充分利用 UB 空间。

- 观察如下流水图，绿色为 Vector 侧的流水，橙色为 Cube 侧的流水，可以看出两侧的流水都存在大段的空隙，CV 之间流水很大程度上未并行，需要考虑 CV 流水优化。

图 6-3 优化前算子流水

- 对于上述场景一，causal 场景下可能存在核间分布不均匀的情况，如下图经过 atten_mask 掩码后，红色部分是算子需要计算的部分，绿色无需计算；如果不按照基本块的个数来分核，按照第一根轴的大小 8（行）来分核，假设平均分到 9 个核上，每个核做 `ceil(8 / 9) = 1` 行，则第 1 个核只需做 1 个基本块，但是第 8 个核需要做 8 个基本块的计算，出现严重的负载不均衡。因此需要考虑将红色块均匀分到多个核上计算。

图 6-4 causal 场景 atten_mask 形状

- 场景一的 Profiling 数据如下，aic_fixpipe_ratio 占比极高，可能存在 FixPipe bound。

图 6-5 场景一 Profiling 数据

- 场景二的 Profiling 数据如下，mte2_ratio 占比高，可能存在 MTE2 bound。

图 6-6 场景二 Profiling 数据

## 设计优化方案

### 优化点一：调整 tiling 基本块

在满足 UB 空间大小够用的情况下，tiling 基本块切分的越大越好。如下图为优化前按照 (64, 128) 切分计算，总共需要循环计算 32 次。

图 6-7 优化前计算基本块及次数

考虑到 UB 空间没有用满，基本块调整到 (128, 128)，如下图优化后只需循环计算 16 次，切分后算子性能提升一倍。

图 6-8 优化后计算基本块及次数

### 优化点二：CV 流水并行

由于 FAG 算子中 Cube 计算比 Vector 计算快且存在依赖性，同时为了减少 CV 之间的通信次数，通过缓存机制实现让 matmul 提前计算多块，这里的缓存机制指的是将 mm 一次性计算多个基本块缓存到 GM 上。如下代码中，SetTail 设置的 singleCoreM 和 singleCoreN 大小分别为 BaseM，BaseN 的倍数，即 matmul 一次发起多个基本块的计算，实现 matmul 结果的缓存，Vector 侧分多次取 matmul 的结果。

```cpp
mm3.SetTail(s2CvExtend, -1, preS1Extend);
mm3.SetTensorA(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN], true);
mm3.SetTensorB(queryGm[mm2aTensorOffsetCv]);
mm3.template IterateAll<false>(dkWorkSpaceGm[bTensorOffsetCv], true);
```

图 6-9 完成 mm1/mm2/mm3 缓存的流水

如上图是实现 mm1、mm2 和 mm3 缓存的流水图，并行度提高，CV 的间隔减小，提升了算子性能。

图 6-10 Vector 等 Cube 流水的间隔插入 Vector 计算

基于缓存 mm1/mm2/mm3 的优化后，在本轮 Vector 计算等 Cube 流水的间隔，插入下一轮循环的 Vector 计算，如上图所示，这样使 Vector 流水与 Cube 流水之间的并行度更高，反映到流水图中为 Vector 计算更密集。原计算过程伪代码与在 CV 间隔中插入下一轮 Vector 计算的伪代码，分别如以下两段所示。

```cpp
// 原计算过程伪代码
// mm1计算;
dropout();
Sub();
// mm2计算;
Softmax();
AttenMask();
...
```

```cpp
// 在Vector等Cube流水的间隔中，插入下一轮循环的Vector计算伪代码
// mm1计算;
dropout();
Sub();
dropout(); // 下一轮循环的Vector计算
Sub(); // 下一轮循环的Vector计算
// mm2计算;
Softmax();
AttenMask();
...
```

### 优化点三：每个核负载均衡

图 6-11 causal 场景优化前每个核计算量

图 6-12 causal 场景优化后每个核计算量

尽量实现每个核的计算量均匀，负载均衡。优化前的分核及每个核的计算量如图 11 causal 场景优化前每个核计算量所示，按照第一根轴的大小 8（行）来分核，平均分到 9 个核上，每个核计算 `ceil(8/9)=1` 行，第 1 个核只计算 1 个基本块，但是第 8 个核计算 8 个基本块。优化后如图 12 causal 场景优化后每个核计算量所示，红色块总共 36 个基本块，均分到每个核上，每个核的计算量为 4 块，性能提升一倍。

### 优化点四：FixPipe 优化

从采集的 Profiling 数据来看，Cube FixPipe 占比高达 81%，出现了很严重的 bound（达到上限）。CAModel 工具打印发现存在很多异常的 128B 搬运，排查代码，发现 workspace 地址未 512B 对齐。

图 6-13 场景一优化前 Profiling 数据

代码实现中使用 SetGlobalBuffer 接口设置 workspace 的起始地址，如果起始地址不是按照 512B 对齐，搬运效率会很低，可以强制 GM 地址 512Byte 对齐来避免这个情况。下面代码中 ADDR_ALIGN_SIZE 即为 512。

```cpp
// init workspace address
syncGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace);
uint64_t workspaceOffsets = SYNC_GLOBAL_WORKSPACE_SIZE;
dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));
workspaceOffsets = (workspaceOffsets + qPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE; dkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));
workspaceOffsets = (workspaceOffsets + kvPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE; dvWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));
workspaceOffsets = (workspaceOffsets + kvPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
// matmul1 and matmul2 workspace size
matmulWorkspaceSize = cubeBaseMN * sizeof(float);
mm1WorkspaceGm.SetGlobalBuffer((__gm__ T2*)(workspace + workspaceOffsets + cBlockIdx * matmulWorkspaceSize)); mm2WorkspaceGm.SetGlobalBuffer((__gm__ T2*)(workspace + workspaceOffsets + coreNum * matmulWorkspaceSize + cBlockIdx * matmulWorkspaceSize)); // drop workspace offset
workspaceOffsets = (workspaceOffsets + coreNum * cubeBaseMN * sizeof(float) * INPUT_NUMS + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
dropWorkSpaceGm.SetGlobalBuffer((__gm__ T1*)workspace + workspaceOffsets / sizeof(T1));
// mul workspace offset
workspaceOffsets = (workspaceOffsets + coreNum * cubeBaseMN * sizeof(half) * 2 + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
mulWorkSpaceGm.SetGlobalBuffer((__gm__ T1*)workspace + workspaceOffsets / sizeof(T1));
```

修改代码，workspace 地址经过 512B 对齐后，FixPipe 时间减半。

图 6-14 场景一优化后 Profiling 数据

### 优化点五：MTE2 优化

结合如下的 Profiling 数据和流水图，可以看出 MTE2 bound，且部分 MTE2 搬运时间异常。

图 6-15 场景二 Profiling 数据

图 6-16 场景二流水图

将输入数据排布格式从 BSH 更改为 BNSD 后，数据搬运连续，不需要跳地址读取数据，搬运效率提升一倍，部分异常搬运时长降低了一半。

## 验证优化方案性能收益

- **调整 tiling 基本块**：理论评估 Vector 切块越大，计算和搬运循环次数越少，同时能够充分利用搬运带宽和 Vector 算力。基本块大小从 (64, 128) 增大到 (128, 128) 后，性能提升一倍，实测与理论分析一致。
- **CV 流水并行**：CV 流水掩盖的时间即为提升的性能，符合预期的收益。
- **核间负载均衡**：优化前负载最多的核计算量减少的倍数，即为预期提升的性能；案例中优化前负载最多的核的计算量大小为 8 块，优化后为 4 块，实际性能提升一倍，符合预期的收益。
- **FixPipe 优化**：从 Profiling 数据看出 FixPipe 占比 0.8，优化后占比 0.55，实测算子性能提升 45%，与理论分析一致。
- **MTE2 优化**：从 Profiling 数据看出 MTE2 占比 0.52，优化后占比减少一半，实测算子性能提升 30%，与理论分析一致。

## 总结

融合算子场景，可参考此优化。
