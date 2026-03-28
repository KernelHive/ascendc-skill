### Matmul 算子优化 Tiling 策略

## 案例介绍

本案例对 Matmul 算子进行性能分析和优化。Matmul 算子实现的功能是矩阵乘法，其中主要包含的流水为数据搬入和搬出流水，Cube 计算流水。

以矩阵维度 M = 4096，N = 5120，K = 4096，输入数据类型 half，输出数据类型 float，输出格式是 ND 为例，性能验证平台为 Atlas A2 训练系列产品/Atlas A2 推理系列产品，介绍针对 Matmul 算子的优化手段，包括优化分核逻辑、优化基本块、使能大包搬运。

- **分核逻辑**：开启尽量多的 Cube 核使能并行计算。
- **优化基本块**：选择最优的 baseM、baseN、baseK 参数，其中 baseM、baseN、baseK 为 Matmul Tiling 中的参数。
- **使能大包搬运**：从 GM 搬运数据到 L1 时，对于 A 矩阵，一次搬入 depthA1 个基本块，基本块大小为 baseM * baseK，对于 B 矩阵，一次搬入 depthB1 个基本块，基本块大小为 baseN * baseK。使能大包搬运后，一次搬入的数据量变大，提升 MTE2 搬运效率。

## 获取性能数据

使用 msProf 工具获取算子的 Profiling 数据，重点分析 MTE2、Cube、Scalar pipeline 的流水情况。

## 分析主要瓶颈点

![图 6-21 优化前 Profiling 数据]()

由以上 Profiling 数据，可以看出 MTE2 耗时占比较大，当前性能瓶颈点在于 MTE2 流水。

- Profiling 数据的 Block Dim 可见分核未分满，考虑分核逻辑的优化。设 CurrentCore 是未优化前分核的 Cube 核数，MaxCore 为最大 Cube 核数，当开启全部核并行做当前 shape 数据量的计算时，预估性能收益为 MaxCore / CurrentCore 的倍数。
- 优化基本块切分，将影响搬运数据的效率，算子搬运的总数据量为搬运的左矩阵和右矩阵数据量之和。在 Matmul 计算 K 方向不能全载的场景下，根据矩阵乘法的算法，搬运左矩阵的次数为 N / baseN，搬运右矩阵的次数为 M / baseM，即搬运总数据量 totalCnt = (N / baseN) * M * K + (M / baseM) * K * N。预估性能收益为搬运数据量的比值，优化前搬运数据量 totalCnt0 / 优化后搬运数据量 totalCnt1，化简后结果为 (1 / baseM0 + 1 / baseN0) / (1 / baseM1 + 1 / baseN1)，其中，baseM0、baseN0 为优化前基本块参数，baseM1、baseN1 为优化后基本块参数。
- 使能大包搬运后，指令条数变化、地址对齐等因素会影响性能，按照经验预估，对于 MTE2 为性能瓶颈的场景，会有 20%+ 的 MTE2 性能收益。

## 设计优化方案

### 优化点一：优化分核逻辑

由 Profiling 数据看出分核数为 4，启动更多的核同时计算，可以提高计算并行度。当前案例使用的 AI 处理器共 20 个核，每个核中包含 1 个 Cube Core 和 2 个 Vector Core。NPU 调用程序中设置 blockDim 为实际使用的核数 20。

```cpp
// 代码片段
uint32_t blockDim = 20; // 优化前 blockDim 为 4
CHECK_ACL(aclInit(nullptr));
int32_t deviceId = 0;
CHECK_ACL(aclrtSetDevice(deviceId));
aclrtStream stream = nullptr;
CHECK_ACL(aclrtCreateStream(&stream));

uint8_t *aHost;
uint8_t *aDevice;
CHECK_ACL(aclrtMallocHost((void **)(&aHost), aFileSize));
CHECK_ACL(
aclrtMalloc((void **)&aDevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
ReadFile("./input/x1_gm.bin", aFileSize, aHost, aFileSize);
// PrintData(aHost, 16, printDataType::HALF);
CHECK_ACL(aclrtMemcpy(aDevice, aFileSize, aHost, aFileSize,
ACL_MEMCPY_HOST_TO_DEVICE));

uint8_t *bHost;
uint8_t *bDevice;
CHECK_ACL(aclrtMallocHost((void **)(&bHost), bFileSize));
CHECK_ACL(
aclrtMalloc((void **)&bDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
ReadFile("./input/x2_gm.bin", bFileSize, bHost, bFileSize);
// PrintData(bHost, 16, printDataType::HALF);
CHECK_ACL(aclrtMemcpy(bDevice, bFileSize, bHost, bFileSize,
ACL_MEMCPY_HOST_TO_DEVICE));

uint8_t *workspaceHost;
uint8_t *workspaceDevice;
CHECK_ACL(aclrtMallocHost((void **)(&workspaceHost), workspaceSize));
CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize,
ACL_MEM_MALLOC_HUGE_FIRST));

uint8_t *tilingHost;
uint8_t *tilingDevice;
CHECK_ACL(aclrtMallocHost((void **)(&tilingHost), tilingFileSize));
CHECK_ACL(aclrtMalloc((void **)&tilingDevice, tilingFileSize,
ACL_MEM_MALLOC_HUGE_FIRST));
CHECK_ACL(aclrtMemcpy(tilingHost, tilingFileSize, GenerateTiling(),
tilingFileSize, ACL_MEMCPY_HOST_TO_HOST));
// PrintData(tilingHost, 16, printDataType::UINT32_T);
CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost,
tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

uint8_t *cHost;
uint8_t *cDevice;
CHECK_ACL(aclrtMallocHost((void **)(&cHost), cFileSize));
CHECK_ACL(
aclrtMalloc((void **)&cDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

// ACLRT_LAUNCH_KERNEL(matmul_custom)
// (blockDim, stream, aDevice, bDevice, cDevice, workspaceDevice, tilingDevice);
matmul_custom_do(blockDim, stream, aDevice, bDevice, cDevice, workspaceDevice, tilingDevice);
```

由于 Matmul API 都是从 Vector 侧发起的，当前案例使用的 AI 处理器中 Cube Core 和 Vector Core 的配比为 1 : 2，所以在 Matmul tiling 计算中需要按照 2 倍的 blockDim 数切分，即 Vector Core 数。NPU 调用程序中设置的实际运行核数是 20 核，所以 Tiling 代码中设置 Tiling API 按照 40 个核进行数据切分，如下代码所示。

```cpp
int usedCoreNum = 40; // 优化前 usedCoreNum 是 8
int runMode = 1;
int32_t baseM = 64; // 64
int32_t baseN = 64; // 64
optiling::TCubeTiling tilingData;
MultiCoreMatmulTiling tilingApi;
tilingApi.SetDim(usedCoreNum);
```

![图 6-22 优化分核逻辑后 Profiling 数据]()

修改代码后，算子执行时间从 12045us 下降到 2532us，约等于 (20核 / 4核) = 5 倍的性能提升。

### 优化点二：优化基本块

当前 Tiling 中设置的 base 块为 [baseM, baseN, baseK] = [64, 64, 256]，这种基本块 Cube 计算 cycle 少，计算访存比（即计算量与需要数据量的比值）低；搬出一次 Matmul 结果到 GM 的 base 块是 64 * 64，由于输出格式是 ND，数据类型是 float，搬出下一次 Matmul 结果的起始地址需要偏移一个 baseN 的大小，即 64 * 4 = 256 字节，导致 fixpipe 搬出时 GM 地址非 512byte 对齐，那么需要设置更优的基本块。

针对当前 shape 较大的场景，基本块的选择原则为计算访存比最大，即在 Cube 计算量最大的情况下，访存的数据量最小。在输入为 fp16 类型的情况下，Cube 执行单元 1 cycle 能算 16 * 16 * 16 个数。根据经验，[baseM, baseN, baseK] = [128, 256, 64] 和 [128, 128, 128] 两种切分方案均满足搬出时 GM 地址 512Byte 对齐（每搬出一次 Matmul 结果时，地址分别偏移 256 * 4byte 和 128 * 4byte），Cube 计算 cycle 数一致，为 (128 * 64 * 256) / (16 * 16 * 16) = (128 * 128 * 128) / (16 * 16 * 16) = 512cycle。

- 针对 [baseM, baseN, baseK] = [128, 256, 64]，计算访存比为 512cycle / (128 * 64 * 2 + 256 * 64 * 2) = 512cycle / 48KB
- 针对 [baseM, baseN, baseK] = [128, 128, 128]，计算访存比为 512cycle / (128 * 128 * 2 + 128 * 128 * 2) = 512cycle / 64KB

可见 [128, 256, 64] 基本块方案的计算访存比更高，计算密度更大，同样的计算量，需要的数据量最小，最大限度提高 Cube 单元的计算量。

修改 Tiling 代码，通过 SetFixSplit() 接口设置 baseM 和 baseN，tiling 函数会自动计算出最优 baseK，这里得到 64。

```cpp
int32_t baseM = 128; // 优化前 baseM 是 64
int32_t baseN = 256; // 优化前 baseN 是 64

optiling::TCubeTiling tilingData;
MultiCoreMatmulTiling tilingApi;
tilingApi.SetDim(usedCoreNum);
tilingApi.SetAType(leftPos, leftFormat, leftDtype, bool(transposeA));
tilingApi.SetBType(rightPos, rightFormat, rightDtype, bool(transposeB));
tilingApi.SetCType(resPos, resFormat, resDtype);
tilingApi.SetBiasType(biasPos, biasFormat, biasDtype);

tilingApi.SetOrgShape(M, N, K);
tilingApi.SetShape(M, N, K);
tilingApi.SetFixSplit(baseM, baseN, -1);
```

使能这组基本块后，MTE2 耗时（对应 aic_mte2_time）从 2452us 降低到 808us，MTE2 性能提升 3 倍。

![图 6-23 优化基本块后 Profiling 数据]()

### 优化点三：使能大包搬运

当前带宽利用率为：totalSize / mte2Time = totalCnt * dtype / mte2Time，代入数据计算为 2491GB/s。未使能大包搬运的情况下，矩阵从 GM 搬运到 L1 一次只搬运 1 个基本块。通过模板参数使能大包搬运，一次搬运多个基本块，提高 MTE2 带宽利用率。

```cpp
// 原始 matmul 对象定义:
Matmul<AscendC::MatmulType<TPosition::GM, CubeFormat::ND, A_T>,
AscendC::MatmulType<TPosition::GM, CubeFormat::ND, B_T>,
AscendC::MatmulType<TPosition::GM, CubeFormat::ND, C_T>,
AscendC::MatmulType<TPosition::GM, CubeFormat::ND, BiasT>>>
mm;

// 通过在定义 matmul 对象的模板参数里加上 CFG_MDL 参数使能大包搬运功能：
Matmul<AscendC::MatmulType<TPosition::GM, CubeFormat::ND, A_T>,
AscendC::MatmulType<TPosition::GM, CubeFormat::ND, B_T>,
AscendC::MatmulType<TPosition::GM, CubeFormat::ND, C_T>,
AscendC::MatmulType<TPosition::GM, CubeFormat::ND, BiasT>, CFG_MDL>>
mm;
```

从下图可以看到，使能大包搬运后，MTE2 耗时从 808us 下降到 591us，带宽利用率代入数据计算为 3406GB/s，利用率提升 36%+，Cube 利用率达到 80%+。

![图 6-24 使能大包搬运后 Profiling 数据]()

## 验证优化方案性能收益

- **优化分核逻辑**：实际收益 4.75 倍，约等于 (20核 / 4核) = 5 倍收益，并且考虑到核的启动开销，可以认为收益一致。
- **优化基本块**：实际收益约 3 倍，理论评估代入上述分析公式，收益为 (1 / 64 + 1 / 64) / (1 / 128 + 1 / 256)，约等于 2.7 倍，考虑到 cache 缓存的影响，认为收益一致。
- **大包搬运**：实际收益 25%+，与经验值一致。

## 总结

优化点一和优化点二的适用场景，需要 shape 足够大，数据量足够多，才能分满核和使能最优的基本块。大 shape 场景下，MTE2 Bound 算子可参考此案例的优化手段。
