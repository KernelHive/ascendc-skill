## 含有 Matmul 高阶 API 的算子精度问题

## 概述

本节针对含有 Matmul 高阶 API 的算子，为排查算子精度问题是否为算子中 Matmul 高阶 API 调用方式导致，提供初步的问题定界和定位指导。如未特殊说明，下面均以 Atlas A2 训练系列产品/Atlas A2 推理系列产品上的案例为例。

具体排查过程主要有如下六个步骤：

1. CPU 域调试，观察报错信息
2. Matmul Tiling 是否有修改，修改是否合理
3. 算子隐藏 Vector 计算，仅调用 Matmul API，算子功能是否正确
4. 单核执行，算子功能是否正确
5. 排查 Matmul API 的使用是否正确
6. 用于算子调测的 golden 脚本是否正确

## 步骤 1：CPU 域调试，观察报错信息

在完成算子代码的开发后，优先通过 Kernel 直调中的 CPU 调测工程，调试算子的功能。在 CPU 域调试时，若编译或执行报错，日志中一般会有明显的报错信息。根据报错信息的提示内容，通常可以快速定位到问题所对应的代码位置。这种方法尤其对 DataCopy 参数设置错误导致的地址越界、算子 Tiling 参数设置不正确、其他内存越界访问等基础参数的使用问题，可以快速定位到具体原因。

### 案例

以下为 matmul 算子核函数的代码片段。该段代码实现了根据 Global Memory 上的 A、B 矩阵和 Tiling 信息，计算每个核要使用数据的地址偏移、创建 Matmul 对象，计算得到 Matmul 结果。

```cpp
extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm)
{
    using A_T = half;
    using B_T = half;
    using C_T = float;

    AscendC::TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    AscendC::GlobalTensor<A_T> aGlobal;
    AscendC::GlobalTensor<B_T> bGlobal;
    AscendC::GlobalTensor<C_T> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(b), tiling.Ka * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(c), tiling.M * tiling.N);

    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;
    bool isTransA = false;
    bool isTransB = true;

    int tailM = 0;
    int tailN = 0;
    CalcGMOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC, tailM, tailN, isTransA, isTransB);

    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC = cGlobal[offsetC];

    AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T>,
                    AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T>,
                    AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_T>> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);

    mm.SetTensorA(gmA, isTransA);
    mm.SetTensorB(gmB, isTransB);
    mm.SetTail(tailM, tailN);
    mm.IterateAll(gmC);
    mm.End();
}
```

以下为上述代码在 CPU 域调试时输出的执行结果。

```
[ASSERT] $HOME/Ascend/CANN-7.5/x86_64-linux/ascendc/include/highlevel_api/lib/matmul/matmul_client.h:268: Assertion `isTransposeB <= B_TYPE::isTrans && "It is not allowed to do B transpose when matmul B transpose is not defined."'
[ASSERT] $HOME/Ascend/CANN-7.5/x86_64-linux/ascendc/include/highlevel_api/lib/matmul/matmul_client.h:268: Assertion `isTransposeB <= B_TYPE::isTrans && "It is not allowed to do B transpose when matmul B transpose is not defined."'
[ERROR][AIV_1][pid 1010818] error happened! =========
SIGABRT Signal (Abort Signal from abort) catched, backtrace info:
[#0] 0x0000000000009cd2: Handler(int) at $HOME/Ascend/latest/tools/tikicpulib/lib/include/kern_fwk.h:106
[#1] 0x00000000000060b7: main at $HOME/samples/Precision_Check_Guide/samples-master/operator/MatmulCustomSample/KernelLaunch/MatmulInvocationNeo-cpu_check/main.cpp:50 (discriminator 126)
[#2] 0x00000000000086de: _start at ??:?

[ERROR][AIV_0][pid 1010817] error happened! =========
SIGABRT Signal (Abort Signal from abort) catched, backtrace info:
[#0] 0x0000000000009cd2: Handler(int) at $HOME/Ascend/latest/tools/tikicpulib/lib/include/kern_fwk.h:106
[#1] 0x00000000000060b7: main at $HOME/samples/Precision_Check_Guide/samples-master/operator/MatmulCustomSample/KernelLaunch/MatmulInvocationNeo-cpu_check/main.cpp:50 (discriminator 126)
[#2] 0x00000000000086de: _start at ??:?
```

本案例中的算子有精度问题，于是使用 CPU 调测该算子功能，CPU 运行后，根据报错信息提示的矩阵 B 的 transpose 未定义，查看矩阵 B 的相关设置代码，发现 Matmul 对象定义时未设置矩阵 B 的 `B_TYPE::isTrans`，而 `SetTensorB` 接口设置了 `isTransB = true`，导致执行报错。所以，此问题的根因为 `SetTensorB` 设置的 `isTransB` 值与 `B_TYPE` 不符。

## 步骤 2：Matmul Tiling 是否有修改，修改是否合理

一般含有 Matmul 的算子 Tiling 实现中，通过调用 `GetTiling` 接口获取 Matmul Tiling，其数据类型为 `TCubeTiling` 结构体，这时这组 Tiling 值是合法的。某些情况下，用户自定义了一组 `TCubeTiling` 参数值，或者，基于 `GetTiling` 接口返回的 `TCubeTiling`，自行修改了其中的部分值，这样的修改需要满足参数间的制约条件。

为获取所有 Tiling 参数值，需要打印 Tiling 参数相关的日志。设置日志环境变量，获取 MatmulTiling 参数值。设置环境变量的命令如下：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

在日志中搜索 "MatmulTiling" 关键字，参照 `TCubeTiling` 约束条件，检查 Tiling 取值是否合法。若不满足某条约束条件，需要修改对应的相关参数，使该组 `TCubeTiling` 参数值均合法。

```bash
cat test_tiling.log | grep MatmulTiling  # test_tiling.log 为示例日志文件名
```

日志输出示例：

```
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.864 [matmul_tiling_base.cpp:697][PrintTilingDataInfo] MatmulTiling: M = 1024
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.870 [matmul_tiling_base.cpp:698][PrintTilingDataInfo] MatmulTiling: N = 640
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.873 [matmul_tiling_base.cpp:699][PrintTilingDataInfo] MatmulTiling: Ka = 256
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.876 [matmul_tiling_base.cpp:700][PrintTilingDataInfo] MatmulTiling: Kb = 256
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.879 [matmul_tiling_base.cpp:701][PrintTilingDataInfo] MatmulTiling: singleCoreM = 512
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.882 [matmul_tiling_base.cpp:702][PrintTilingDataInfo] MatmulTiling: singleCoreN = 640
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.884 [matmul_tiling_base.cpp:703][PrintTilingDataInfo] MatmulTiling: singleCoreK = 256
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.887 [matmul_tiling_base.cpp:704][PrintTilingDataInfo] MatmulTiling: baseM = 256
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.890 [matmul_tiling_base.cpp:705][PrintTilingDataInfo] MatmulTiling: baseN = 128
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.893 [matmul_tiling_base.cpp:706][PrintTilingDataInfo] MatmulTiling: baseK = 64
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.896 [matmul_tiling_base.cpp:707][PrintTilingDataInfo] MatmulTiling: depthA1 = 10
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.899 [matmul_tiling_base.cpp:708][PrintTilingDataInfo] MatmulTiling: depthB1 = 2
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.902 [matmul_tiling_base.cpp:709][PrintTilingDataInfo] MatmulTiling: depthAL1CacheUB = 0
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.905 [matmul_tiling_base.cpp:710][PrintTilingDataInfo] MatmulTiling: depthBL1CacheUB = 0
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.908 [matmul_tiling_base.cpp:711][PrintTilingDataInfo] MatmulTiling: stepM = 2
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.912 [matmul_tiling_base.cpp:712][PrintTilingDataInfo] MatmulTiling: stepN = 1
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.915 [matmul_tiling_base.cpp:713][PrintTilingDataInfo] MatmulTiling: isBias = 1
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.917 [matmul_tiling_base.cpp:714][PrintTilingDataInfo] MatmulTiling: transLength = 0
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.920 [matmul_tiling_base.cpp:715][PrintTilingDataInfo] MatmulTiling: iterateOrder = 0
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.923 [matmul_tiling_base.cpp:716][PrintTilingDataInfo] MatmulTiling: shareMode = 0
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.926 [matmul_tiling_base.cpp:717][PrintTilingDataInfo] MatmulTiling: usedL1Size = 295424
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.929 [matmul_tiling_base.cpp:718][PrintTilingDataInfo] MatmulTiling: usedL0CSize = 131072
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.932 [matmul_tiling_base.cpp:719][PrintTilingDataInfo] MatmulTiling: usedUBSize = 0
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.935 [matmul_tiling_base.cpp:720][PrintTilingDataInfo] MatmulTiling: batchM = 1
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.938 [matmul_tiling_base.cpp:721][PrintTilingDataInfo] MatmulTiling: batchN = 1
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.941 [matmul_tiling_base.cpp:722][PrintTilingDataInfo] MatmulTiling: singleBatchM = 1
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.943 [matmul_tiling_base.cpp:723][PrintTilingDataInfo] MatmulTiling: singleBatchN = 1
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.946 [matmul_tiling_base.cpp:724][PrintTilingDataInfo] MatmulTiling: stepKa = 4
[INFO] ASCENDCKERNEL(1202803,ascendc_kernels_bbit):2024-10-12-08:53:59.636.949 [matmul_tiling_base.cpp:725][PrintTilingDataInfo] MatmulTiling: stepKb = 1
```

例如，根据如上打印出的 `TCubeTiling` 参数，对照 `TCubeTiling` 约束条件查看各个参数的取值，`depthA1` 的取值应该等于 `stepM * stepKa` 或者 `stepM * stepKa * 2`，而 `depthA1` 的取值为 10，既不等于 `stepM * stepKa = 8`，也不等于 `stepM * stepKa * 2 = 16`，不满足约束条件，因此需要校正 `depthA1` 的值。

## 步骤 3：算子隐藏 Vector 计算，仅调用 Matmul API，算子功能是否正确

融合算子的代码既包含 Matmul API，也包含 Vector 计算 API。通过在算子代码中删除 Vector 计算 API，只保留 Matmul API，快速定界是否为 Matmul API 的错误使用导致了融合算子的精度问题。具体排查过程为：修改算子代码逻辑，删除 Vector 计算的代码，同步完成 golden 脚本相应修改，完成适配修改后，CPU 域或 NPU 域上执行算子，观察算子结果是否正确。若算子结果正确，说明代码中 Matmul API 的使用方式正确，需要继续排查 Vector 计算是否正确；反之，若算子结果不正确，需要继续排查 Matmul API 的使用是否正确。

### 案例

以融合算子 `matmul_leakyrelu` 为例，执行算子后，出现如下图所示的精度问题。

```
data index: 000195, expected: -0.693000019, actual: -69.300003052, rdiff: -99.000000
data index: 000196, expected: -0.209000006, actual: -20.899999619, rdiff: -99.000000
data index: 000197, expected: -0.517000020, actual: -51.700000763, rdiff: -99.000000
data index: 000200, expected: -0.193000004, actual: -19.300001144, rdiff: -99.000000
data index: 000202, expected: -0.684000015, actual: -68.400001526, rdiff: -99.000000
data index: 000204, expected: -0.422000021, actual: -42.200000763, rdiff: -98.999992
data index: 000209, expected: -0.109000005, actual: -10.900000572, rdiff: -99.000000
error ratio: 0.4517, tolerance: 0.0001
[ERROR] result error
```

修改算子代码，注释屏蔽 LeakyRelu API 计算，同时，需要适配修改相应的内存分配和涉及的同步等代码；然后，注释 golden 脚本中 LeakyRelu 计算，具体修改示例如下。

以下代码为算子核函数的代码片段。

```cpp
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Process(AscendC::TPipe *pipe)
{
    uint32_t computeRound = 0;

    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);
    while (matmulObj.template Iterate<true>()) {
        MatmulCompute();
        // LeakyReluCompute(); // 注释 LeakyReluCompute Vector 计算
        CopyOut(computeRound);
        computeRound++;
    }
    matmulObj.End();
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::MatmulCompute()
{
    reluOutLocal = reluOutQueue_.AllocTensor<cType>();
    matmulObj.template GetTensorC<true>(reluOutLocal, false, true);
    reluOutQueue_.EnQue(reluOutLocal); // 将 LeakyReluCompute() 接口里的 reluOutLocal 结果输出提前到这里
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::LeakyReluCompute()
{
    LeakyRelu(reluOutLocal, reluOutLocal, (cType)0.1, tiling.baseM * tiling.baseN);
    reluOutQueue_.EnQue(reluOutLocal);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::CopyOut(uint32_t count)
{
    reluOutQueue_.DeQue<cType>();
    const uint32_t roundM = tiling.singleCoreM / tiling.baseM;
    const uint32_t roundN = tiling.singleCoreN / tiling.baseN;
    uint32_t startOffset = (count % roundM * tiling.baseM * tiling.N + count / roundM * tiling.baseN);
    AscendC::DataCopyParams copyParam = {(uint16_t)tiling.baseM, (uint16_t)(tiling.baseN * sizeof(cType) / AscendC::DEFAULT_C0_SIZE), 0,
                                        (uint16_t)((tiling.N - tiling.baseN) * sizeof(cType) / AscendC::DEFAULT_C0_SIZE)};
    DataCopy(cGlobal[startOffset], reluOutLocal, copyParam);
    reluOutQueue_.FreeTensor(reluOutLocal);
}
```

以下代码为 golden 生成脚本的代码片段。

```python
def gen_golden_data():
    M = 1024
    N = 640
    K = 256

    input_a = np.random.randint(-10, 10, [M, K]).astype(np.float16)
    input_b = np.random.randint(-10, 10, [K, N]).astype(np.float16)
    input_bias = np.random.randint(-10, 10, [N]).astype(np.float32)
    alpha = 0.001
    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)
    # golden = np.where(golden >= 0, golden, golden * alpha) # 与 kernel 保持一致，golden 生成也需注释相应的 LeakyRelu 计算
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_a.tofile("./input/x1_gm.bin")
    input_b.tofile("./input/x2_gm.bin")
    input_bias.tofile("./input/bias.bin")
    golden.tofile("./output/golden.bin")
```

删除 LeakyRelu 计算后，执行用例，算子结果比对正确，如下所示。

```
-- Installing: $HOME/samples/Precision_Check_Guide/samples-master/operator/MatmulLeakyReluCustomSample/KernelLaunch/MatmulLeakyReluInvocation_cube_vec/out/bin/ascendc_kernels_bbit
8901941eee314bcd64d24ff5f8d21247 output/golden.bin
8901941eee314bcd64d24ff5f8d21247 output/output.bin
error ratio: 0.0000, tolerance: 0.0001
test pass
```

由此可确定，算子代码中已正确使用 Matmul API，并得到了正确的 Matmul API 计算结果，需要继续定位 `LeakyReluCompute` 函数内 `LeakyRelu` 接口使用中存在的问题。

## 步骤 4：单核执行，算子功能是否正确

验证单核场景下算子的功能是否正确，可以帮助快速定界是 Matmul API 的计算结果不符合预期，还是算子代码中错误调用 Matmul API 导致。由于 Matmul API 内部实现的是单核的计算逻辑，所以单核的计算结果正确，而多核的计算结果错误的情况，说明单核上的 Matmul API 的使用及计算正确，这时需要排查与多核切分相关的代码逻辑是否正确，比如每个核的输入和输出地址偏移是否正确，每个核上的尾块地址设置是否正确。如果验证单核场景下，算子精度不正确，需要排查 Matmul API 的使用是否正确，具体可参考步骤 5。

提示，包含 Matmul 的算子的 Tiling 实现中，Matmul 的多核 Tiling 需要使用 `MultiCoreMatmulTiling` 构造多核 Tiling 对象，通过 `SetDim` 接口设置 Matmul 计算所用的核数。注意：这里设置的核数为 Matmul 计算所用的核数，仅在多核场景下设置，用于计算 tiling 参数。如下两个案例为 MIX 模式的算子，`SetDim` 的设置规则请参考 MIX 场景核数设置规则。

### 案例 1：多核切分场景，输出地址偏移不正确

以 M=512，N=1024，K=512 的 Matmul 为例，MIX 模式的算子代码中设置 AIC 核数为 4，AIV 核数为 8，因为本案例以分离模式为例，所以 `SetDim` 设置为 AIV 核数的取值 8。多核场景下执行该算子，计算结果精度错误。

以下为算子 Tiling 计算的代码片段。

```cpp
uint8_t *GenerateTiling(const char *socVersion)
{
    int M = 512;
    int N = 1024;
    int K = 512;

    TPosition leftPosition = TPosition::GM;
    CubeFormat leftFormat = CubeFormat::ND;
    DataType leftDtype = DataType::DT_FLOAT16;
    bool isTransA = false;

    TPosition rightPosition = TPosition::GM;
    CubeFormat rightFormat = CubeFormat::ND;
    DataType rightDtype = DataType::DT_FLOAT16;
    bool isTransB = false;

    TPosition resultPosition = TPosition::GM;
    CubeFormat resultFormat = CubeFormat::ND;
    DataType resultDtype = DataType::DT_FLOAT;

    bool isBias = false;

    int usedCoreNum = 8;
    int32_t baseM = 128;
    int32_t baseN = 256;

    optiling::TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    tilingApi.SetDim(usedCoreNum); // 设置为 AIV 核数 8
    tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
    tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
    tilingApi.SetCType(resultPosition, resultFormat, resultDtype);

    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetFixSplit(baseM, baseN, -1);
    tilingApi.SetBias(isBias);
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t res = tilingApi.GetTiling(tilingData);
    if (res == -1) {
        std::cout << "gen tiling failed" << std::endl;
    }
    return GetTilingBuf(&tilingData);
}
```

以下为算子核函数的代码片段。

```cpp
__aicore__ inline void CalcGMOffset(int blockIdx, const TCubeTiling &tiling, int &offsetA, int &offsetB, int &offsetC,
                                    int &tailM, int &tailN, bool isTransA, bool isTransB)
{
    uint32_t mSingleBlocks = CeilDiv(tiling.M, tiling.singleCoreM);
    uint32_t mCoreIndx = blockIdx % mSingleBlocks;
    uint32_t nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    if (isTransA) {
        offsetA = mCoreIndx * tiling.singleCoreM;
    }
    offsetB = nCoreIndx * tiling.singleCoreN;
    if (isTransB) {
        offsetB = nCoreIndx * tiling.Kb * tiling.singleCoreN;
    }
    offsetC = mCoreIndx * tiling.singleCoreN * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN; // 此处的 tiling.singleCoreN 参数错误，应为 tiling.N

    tailM = tiling.M - mCoreIndx * tiling.singleCoreM;
    tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;

    tailN = tiling.N - nCoreIndx * tiling.singleCoreN;
    tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
}

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm)
{
    using A_T = half;
    using B_T = half;
    using C_T = float;

    AscendC::TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    AscendC::GlobalTensor<A_T> aGlobal;
    AscendC::GlobalTensor<B_T> bGlobal;
    AscendC::GlobalTensor<C_T> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(b), tiling.Ka * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(c), tiling.M * tiling.N);

    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;
    bool isTransA = false;
    bool isTransB = false;

    int tailM = 0;
    int tailN = 0;
    CalcGMOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC, tailM, tailN, isTransA, isTransB);

    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC = cGlobal[offsetC];

    AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T>,
                    AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T>,
                    AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_T>> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);

    mm.SetTensorA(gmA, isTransA);
    mm.SetTensorB(gmB, isTransB);
    mm.SetTail(tailM, tailN);
    mm.IterateAll(gmC);
    mm.End();
}
```

执行算子，精度校验失败：

```
data index: 000609, expected: 12979.000000000, actual: 0.000000000, rdiff: 1.000000
data index: 000610, expected: 12931.000000000, actual: 0.000000000, rdiff: 1.000000
data index: 000611, expected: 13120.000000000, actual: 0.000000000, rdiff: 1.000000
data index: 000612, expected: 12275.000000000, actual: 0.000000000, rdiff: 1.000000
error ratio: 0.8750, tolerance: 0.0001
[ERROR] result error
```

修改测试脚本和算子 Tiling 的代码，通过验证单核上的算子执行结果，快速定界。具体如下：

修改算子调测代码，只启动单核，CPU 调测代码中将 `ICPU_RUN_KF` 宏接口中的 `blockDim` 设置为 1（表示一组 AIC 和 AIV）；算子的 Tiling 实现中，设置单核场景，AIC 核数为 1，AIV 核数为 2，`SetDim` 设置为 AIV 核数的取值 2。如下代码所示。

以下为调测脚本的代码片段。

```cpp
uint32_t blockDim = 1;
ICPU_RUN_KF(matmul_custom, blockDim, a, b, c, workspace, tiling);
```

以下为算子 Tiling 计算的代码片段。

```cpp
int usedCoreNum = 2;
tilingApi.SetDim(usedCoreNum);
```

修改为单核场景后，执行算子：

```
-- Installing: $HOME/samples/Precision_Check_Guide/samples-master/operator/MatmulCustomSample/KernelLaunch/MatmulInvocationNeo-muticore/out/bin/ascendc_kernels_bbit
efaf4dc1e484bc3778cac65f56244e59 output/golden.bin
efaf4dc1e484bc3778cac65f56244e59 output/output.bin
error ratio: 0.0000, tolerance: 0.0001
test pass
```

从上述比对结果可看出，单核验证结果正确，此时可以定界导致精度的问题与多核逻辑相关。

首先排查多核切分后的输入和输出地址偏移。分析 `CalcGMOffset` 函数，定位到矩阵 C 的偏移地址 `offsetC` 计算错误，正确的偏移应该是 `mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN`。将 `offsetC` 修改为正确的偏移地址后，执行算子，结果比对正确。

提示，在上述单核场景的修改验证中，AIC 核数为 1，AIV 核数为 2；若想进一步验证，不引入任何多核切分，AIC 核数和 AIV 核数均修改为 1，代码修改示例如下：

- 在核函数中 `REGIST_MATMUL_OBJ` 接口后，利用判断代码，BlockIdx 不为 0 的 AIV 核退出。

以下为算子核函数的代码片段。

```cpp
extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm)
{
    using A_T = half;
    using B_T = half;
    using C_T = float;

    AscendC::TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    AscendC::GlobalTensor<A_T> aGlobal;
    AscendC::GlobalTensor<B_T> bGlobal;
    AscendC::GlobalTensor<C_T> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(re
