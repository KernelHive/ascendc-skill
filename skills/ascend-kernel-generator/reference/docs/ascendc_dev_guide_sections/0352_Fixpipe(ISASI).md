##### Fixpipe(ISASI)

```markdown
## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | 仅支持包含FixpipeParamsV220参数的接口 |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | 仅支持包含FixpipeParamsV220参数的接口 |
| Atlas 200I/500 A2 推理产品 | 仅支持包含FixpipeParamsM300参数的接口 |
| Atlas 推理系列产品AI Core | × |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

矩阵计算完成后，对结果进行处理，例如对计算结果进行量化操作，并把数据从CO1搬迁到Global Memory中。

## 函数原型

### 传入FixpipeParamsV220

- **通路CO1->GM，不使能tensor量化功能：**
```cpp
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src, const FixpipeParamsV220& intriParams)
```

- **通路CO1->GM，使能tensor量化功能：**
```cpp
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR,
typename S = uint64_t, typename Std::enable_if<Std::is_same<PrimT<S>, uint64_t>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src, const LocalTensor<S>& cbufWorkspace, const FixpipeParamsV220& intriParams);
```

### 传入FixpipeParamsM300

- **通路CO1->UB，不使能tensor量化功能：**
```cpp
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src, const FixpipeParamsM300& intriParams)
```

- **通路CO1->UB，使能tensor量化功能：**
```cpp
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR,
typename S = uint64_t, typename Std::enable_if<Std::is_same<PrimT<S>, uint64_t>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src, const LocalTensor<S>& cbufWorkspace, const FixpipeParamsM300& intriParams);
```

- **通路CO1->GM，不使能tensor量化功能：**
```cpp
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src, const FixpipeParamsM300& intriParams)
```

- **通路CO1->GM，使能tensor量化功能：**
```cpp
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR,
typename S = uint64_t, typename Std::enable_if<Std::is_same<PrimT<S>, uint64_t>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src, const LocalTensor<S>& cbufWorkspace, const FixpipeParamsM300& intriParams)
```

## 参数说明

### 表 15-332 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 目的操作数数据类型 |
| U | 源操作数数据类型 |
| config | Fixpipe相关配置参数，类型为FixpipeConfig。取值如下：<br>• CFG_ROW_MAJOR（默认取值）：使能NZ2ND，输出数据格式为ND格式<br>• CFG_NZ: 不使能NZ2ND，输出数据格式为NZ格式 |
| S | cbufWorkspace的数据类型<br>• 当目的操作数、源操作数、cbufWorkspace使用基础数据类型时，模板参数S必须为uint64_t类型，否则编译失败<br>• 当目的操作数、源操作数、cbufWorkspace使用TensorTrait类型时，模板参数S的LiteType必须为uint64_t类型，否则编译失败<br>模板参数S后一个模板参数仅用于上述数据类型检查，用户无需关注 |

```cpp
struct FixpipeConfig {
    CO2Layout format;
};

enum class CO2Layout : uint8_t {
    NZ = 0,        // 不使能NZ2ND，输出数据格式仍为NZ格式
    ROW_MAJOR,     // 使能NZ2ND，输出数据格式为ND格式
};

constexpr FixpipeConfig CFG_NZ = {CO2Layout::NZ};
constexpr FixpipeConfig CFG_ROW_MAJOR = {CO2Layout::ROW_MAJOR};
```

### 表 15-333 参数说明

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| dst | 输出 | 目的操作数，类型为GlobalTensor<br>• 针对GlobalTensor：<br>&nbsp;&nbsp;Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：int8_t、uint8_t、half、bfloat16_t、int32_t、float<br>&nbsp;&nbsp;Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：int8_t、uint8_t、half、bfloat16_t、int32_t、float<br>&nbsp;&nbsp;Atlas 200I/500 A2 推理产品，支持的数据类型为：int8_t、uint8_t、half、bfloat16_t、int32_t、float<br>数据格式为NZ或ND格式。经过Fixpipe处理，在量化操作之后，会将矩阵计算中多申请的数据删除 |
| src | 输入 | 源操作数，支持的TPosition为CO1，为Mmad接口计算的结果，类型为LocalTensor数据结构的定义请参考15.1.3.1 LocalTensor。支持的数据类型为float/int32_t，支持的TPosition为CO1，数据格式为NZ格式。起始地址需要满足64B对齐 |
| intriParams | 输入 | Fixpipe搬运参数，具体定义请参考`${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_fixpipe.h`，`${INSTALL_DIR}`请替换为CANN软件安装后文件存储路径。参数说明请参考表15-334 |
| cbufWorkspace | 输入 | 量化参数，类型为LocalTensor<uint64_t>，支持的TPosition为A1。仅当quantPre为VDEQF16/VQF322B8_PRE/VREQ8时支持，quantPre介绍请参考FixpipeParamsV220/FixpipeParamsM300/FixpipeParamsM310结构体中quantPre部分 |

### 表 15-334 Fixpipe 搬运参数结构体说明

| 参数名称 | 数据类型 | 含义 |
|----------|----------|------|
| nSize | 输入 | 源NZ矩阵在N方向上的大小<br>• 不使能NZ2ND功能<br>&nbsp;&nbsp;若使能channelSplit功能，nSize必须为8的倍数，取值范围：nSize∈[1, 4095]<br>&nbsp;&nbsp;若不使能channelSplit功能，nSize必须为16的倍数，取值范围：nSize∈[1, 4095]<br>• 使能NZ2ND功能<br>&nbsp;&nbsp;nSize取值范围 ∈[1, 4095] |
| mSize | 输入 | 源NZ矩阵在M方向上的大小<br>• 不使能NZ2ND功能<br>&nbsp;&nbsp;取值范围：mSize∈[1, 65535]<br>• 使能NZ2ND功能<br>&nbsp;&nbsp;取值范围：mSize∈[1, 8192] |
| srcStride | 输入 | 源NZ矩阵中相邻Z排布的起始地址偏移，取值范围：srcStride∈[0, 65535]，单位：C0_Size(16*sizeof(T)，T为src的数据类型) |
| dstStride | 输入 | • 不使能NZ2ND功能<br>&nbsp;&nbsp;目的NZ矩阵中相邻Z排布的起始地址偏移，取值不为0，单位：datablock(32Bytes)<br>• 使能NZ2ND功能<br>&nbsp;&nbsp;目的ND矩阵每一行中的元素个数，取值不为0，单位：element |
| quantPre | 输入 | QuantMode_t是一个枚举类型，用于控制量化模式，默认值为QuantMode_t::NoQuant，即不使能量化功能。QuantMode_t取值如下：<br>• NoQuant，不使能量化功能<br>• F322F16，float量化成half，量化结果支持INF_NAN模式<br>• F322BF16，float量化成bfloat16_t，量化结果支持INF_NAN模式<br>• DEQF16，int32_t量化成half, scalar量化，量化结果不支持INF_NAN模式<br>• VDEQF16，int32_t量化成half，tensor量化，量化结果不支持INF_NAN模式<br>• QF322B8_PRE，float量化成uint8_t/int8_t，scalar量化<br>• VQF322B8_PRE，float量化成uint8_t/int8_t，tensor量化<br>• REQ8，int32_t量化成uint8_t/int8_t，scalar量化<br>• VREQ8，int32_t量化成uint8_t/int8_t，tensor量化 |
| deqScalar | 输入 | scalar量化参数，表示单个scale值，quantPre量化模式为scalar量化时需要设置该参数。支持的数据类型为uint64_t |
| ndNum | 输入 | 源NZ矩阵的数目，也就是传输ND矩阵的数目，取值范围：ndNum∈[1, 65535] |
| srcNdStride | 输入 | 不同NZ矩阵起始地址之间的间隔，取值范围：srcNdStride∈[1, 512]，单位：1024B。当ndNum配置为1时，srcNdStride配置为0即可，不生效 |
| dstNdStride | 输入 | 目的相邻ND矩阵起始地址之间的偏移，取值范围：dstNdstride∈[1, 65535]，单位：element。当ndNum配置为1时，dstNdStride配置为0即可，不生效 |
| reluEn | 输入 | 是否使能relu的开关，false：不使能relu功能；true：使能relu功能 |
| unitFlag | 输入 | 预留参数，用户无需关心，使用默认值0即可 |
| isChannelSplit | 输入 | 是否使能通道拆分的功能。默认为false，不使能该功能。仅在src和dst都为float时才能使能通道拆分，且不能同时使能ChannelSplit和NZ2ND功能 |

## 参数设置示例

### 不使能NZ2ND的情况下

参数设置示例（通过Fixpipe接口搬运并去除dummy数据）和解释说明如下：

- **nSize = 48**：表示源NZ矩阵中待搬运矩阵（图中蓝色区域）在N方向上的大小为48个元素
- **mSize = 24**：表示源NZ矩阵中待搬运矩阵在M方向上的大小为24个元素
- **srcStride = 64**：表示源NZ矩阵中待搬运矩阵相邻Z排布的起始地址偏移，即下图中第一个蓝色Z排布的起始地址与第二个蓝色Z排布的起始地址之间的间隔为64 * C0_Size
- **dstStride = 40**：表示目的NZ矩阵中相邻Z排布的起始地址偏移，即下图中第一个蓝色Z排布的起始地址与第二个蓝色Z排布的起始地址之间的间隔为40 * 32B

**图 15-22 不使能 NZ2ND 参数设置示意图**

### 使能NZ2ND的情况下

参数设置示例和解释说明如下：

- **ndNum = 2**：表示源NZ矩阵的数目为2。图中蓝色区域为NZ矩阵1，紫色区域为NZ矩阵2
- **nSize = 32**：表示源NZ矩阵（图中蓝色区域）在N方向上的大小为32个元素
- **mSize = 48**：表示源NZ矩阵在M方向上的大小为48个元素
- **srcStride = 64**：表示源NZ矩阵中相邻Z排布的起始地址偏移，即下图中第一个蓝色Z排布的起始地址与第二个蓝色Z排布的起始地址之间的间隔为64 * C0_Size
- **dstStride = 64**：表示目的ND矩阵每一行中的元素个数为64
- **srcNdStride = 16**：表示不同NZ矩阵起始地址之间的间隔为16 *1024B
- **dstNdStride = 4096**：表示目的相邻ND矩阵起始地址之间的偏移为4096个元素

**图 15-23 使能 NZ2ND 参数设置示意图**

## 约束说明

- ndNum=0 表示不执行，此指令将不被执行并报warning
- 对于量化输入为float32数据类型的说明如下：
  - 标准的IEEE 754 float32格式为：1bit符号位，8bits指数位，23bits尾数位；当前AI处理器支持的float32格式为：1bit符号位，8bits指数位，10bits尾数位
  - 如果用户提供的是标准的IEEE 754 float32输入，API内部会处理成处理器支持的float32格式进行计算，此时如果golden数据生成过程中使用的是标准的IEEE 754 float32数据，则可能引入精度不匹配问题，需要修正golden数据的生成，将量化参数的23bits尾数位的低13bits数据位清零再参与量化计算

## 调用示例

### 示例一：通路CO1->GM，不使能tensor量化功能接口

输入A矩阵和B矩阵的数据类型为half，输出C矩阵为half，默认配置使能Nz2Nd的格式转换，使能F322F16量化将mmad计算出的结果由float量化成half。

```cpp
#ifdef ASCENDC_CPU_DEBUG
#include "tikicpulib.h"
#endif
#include "kernel_operator.h"

template <typename C_T, typename A_T, typename B_T, typename dstCO1_T>
class KernelMatmul {
public:
    __aicore__ inline KernelMatmul(uint16_t mIn, uint8_t kIn, uint8_t nIn)
    {
        m = mIn;
        k = kIn;
        n = nIn;
        aSize = m * k;
        bSize = k * n;
        cSize = m * n;
        mBlocks = m / AscendC::BLOCK_CUBE;
        nBlocks = n / AscendC::BLOCK_CUBE;
        kBlocks = k / (AscendC::ONE_BLK_SIZE / sizeof(A_T));
    }
    
    __aicore__ inline void Init(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *c)
    {
        aGM.SetGlobalBuffer((__gm__ A_T *)a);
        bGM.SetGlobalBuffer((__gm__ B_T *)b);
        cGM.SetGlobalBuffer((__gm__ C_T *)c);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(A_T));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(A_T));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(B_T));
        pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(B_T));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(dstCO1_T));
    }
    
    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();
        SplitB();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<A_T> a1Local = inQueueA1.AllocTensor<A_T>();
        AscendC::LocalTensor<B_T> b1Local = inQueueB1.AllocTensor<B_T>();

        AscendC::Nd2NzParams dataCopyA1Params;
        dataCopyA1Params.ndNum = 1;
        dataCopyA1Params.nValue = m;
        dataCopyA1Params.dValue = k;
        dataCopyA1Params.srcNdMatrixStride = 0;
        dataCopyA1Params.srcDValue = k;
        dataCopyA1Params.dstNzC0Stride = m;
        dataCopyA1Params.dstNzNStride = 1;
        dataCopyA1Params.dstNzMatrixStride = 0;

        AscendC::Nd2NzParams dataCopyB1Params;
        dataCopyB1Params.ndNum = 1;
        dataCopyB1Params.nValue = k;
        dataCopyB1Params.dValue = n;
        dataCopyB1Params.srcNdMatrixStride = 0;
        dataCopyB1Params.srcDValue = n;
        dataCopyB1Params.dstNzC0Stride = k;
        dataCopyB1Params.dstNzNStride = 1;
        dataCopyB1Params.dstNzMatrixStride = 0;

        // AscendC::DataCopy GM->L1:ND->大N小z
        AscendC::DataCopy(a1Local, aGM, dataCopyA1Params);
        AscendC::DataCopy(b1Local, bGM, dataCopyB1Params);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
    }
    
    __aicore__ inline void SplitA()
    {
        AscendC::LocalTensor<A_T> a1Local = inQueueA1.DeQue<A_T>();
        AscendC::LocalTensor<A_T> a2Local = inQueueA2.AllocTensor<A_T>();
        
        // AscendC::LoadData L1->L0A
        AscendC::LoadData2dParams loadL0AParams;
        loadL0AParams.repeatTimes = mBlocks;
        loadL0AParams.srcStride = 1;
        loadL0AParams.dstGap = kBlocks - 1;
        loadL0AParams.ifTranspose = false;
        
        for (int i = 0; i < kBlocks; i++) {
            AscendC::LoadData(a2Local[i * 16 * (32 / sizeof(A_T))], a1Local[i * m * (32 / sizeof(A_T))], loadL0AParams);
        }
        
        inQueueA2.EnQue<A_T>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }
    
    __aicore__ inline void SplitB()
    {
        AscendC::LocalTensor<B_T> b1Local = inQueueB1.DeQue<B_T>();
        AscendC::LocalTensor<B_T> b2Local = inQueueB2.AllocTensor<B_T>();

        // Load2d transpose L1->L0B
        AscendC::LoadData2dTransposeParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.srcStride = 1;
        loadDataParams.addrMode = 0;
        loadDataParams.repeatTimes = k * n / B32_B16_SIZE;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = n / n_block - 1;
        
        AscendC::LoadDataWithTranspose(b2Local, b1Local, loadDataParams);
        inQueueB1.FreeTensor(b1Local);
        inQueueB2.EnQue<B_T>(b2Local);
    }
    
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<A_T> a2Local = inQueueA2.DeQue<A_T>();
        AscendC::LocalTensor<B_T> b2Local = inQueueB2.DeQue<B_T>();
        AscendC::LocalTensor<dstCO1_T> c1Local = outQueueCO1.AllocTensor<dstCO1_T>();
        
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        
        AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams); // m*n
        outQueueCO1.EnQue<dstCO1_T>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }
    
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<dstCO1_T> c1Local = outQueueCO1.DeQue<dstCO1_T>();
        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = m;
        fixpipeParams.dstStride = n;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 2;
        fixpipeParams.dstNdStride = m*n;
        fixpipeParams.quantPre = QuantMode_t::F322F16;
        
        AscendC::Fixpipe(cGM, c1Local, fixpipeParams);
        outQueueCO1.FreeTensor(c1Local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;
    AscendC::GlobalTensor<A_T> aGM;
    AscendC::GlobalTensor<B_T> bGM;
    AscendC::GlobalTensor<C_T> cGM;
    uint16_t m, k, n;
    uint16_t B32_B16_SIZE = 16 * 16;
    uint8_t n_block = 16;
    uint16_t aSize, bSize, cSize, mBlocks, nBlocks, kBlocks;
};

#define KERNEL_MATMUL(c_type, a_type, b_type, co1_type, mIn, kIn, nIn) \
extern "C" __global__ __aicore__ void cube_matmul_loaddata_operator( \
    __gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *c) \
{ \
    if (g_coreType == AscendC::AIV) { \
        return; \
    } \
    KernelMatmul<c_type, a_type, b_type, co1_type> op(mIn, kIn, nIn);\
    op.Init(a, b, c); \
    op.Process(); \
}

KERNEL_MATMUL(half, half, half, float, 32, 32, 16);
```

**示例结果：**

输入数据A矩阵、B矩阵和输出数据C矩阵（具体数值略）

### 示例二：通路CO1->GM，使能tensor量化功能接口

输入A矩阵和B矩阵的数据类型为int8，输出C矩阵为half，默认配置使能Nz2Nd的格式转换，使能tensor量化（VDEQF16）将mmad计算出的结果由int32量化成half。

```cpp
#ifdef ASCENDC_CPU_DEBUG
#include "tikicpulib.h"
#endif
#include "kernel_operator.h"

template <typename c_T, typename a_T, typename b_T, typename dstCO1_T>
class KernelMatmul {
public:
    __aicore__ inline KernelMatmul(uint16_t mIn, uint8_t kIn, uint8_t nIn)
    {
        m = mIn;
        k = kIn;
        n = nIn;
        aSize = m * k;
        bSize = k * n;
        cSize = m * n;
        mBlocks = m / AscendC::BLOCK_CUBE;
        nBlocks = n / AscendC::BLOCK_CUBE;
        kBlocks = k / (AscendC::ONE_BLK_SIZE / sizeof(a_T));
        deqTensorLen = n;
    }
    
    __aicore__ inline void Init(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *c, __gm__ uint8_t *deqTensor)
    {
        aGM.SetGlobalBuffer((__gm__ a_T *)a);
        bGM.SetGlobalBuffer((__gm__ b_T *)b);
        cGM.SetGlobalBuffer((__gm__ c_T *)c);
        deqTensorGM.SetGlobalBuffer((__gm__ uint64_t *)deqTensor);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(a_T));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(a_T));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(b_T));
        pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(b_T));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(dstCO1_T));
        pipe.InitBuffer(deqQueue, 1, deqTensorLen * sizeof(uint64_t));
    }
    
    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();
        SplitB();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<a_T> a1Local = inQueueA1.AllocTensor<a_T>();
        AscendC::LocalTensor<b_T> b1Local = inQueueB1.AllocTensor<b_T>();
        AscendC::LocalTensor<uint64_t> deqLocal = deqQueue.AllocTensor<uint64_t>();

        AscendC::Nd2NzParams dataCopyA1Params;
        dataCopyA1Params.ndNum = 1;
        dataCopyA1Params.nValue = m;
        dataCopyA1Params.dValue = k;
        dataCopyA1Params.srcNdMatrixStride = 0;
        dataCopyA1Params.srcDValue = k;
        dataCopyA1Params.dstNzC0Stride = m;
        dataCopyA1Params.dstNzNStride = 1;
        dataCopyA1Params.dstNzMatrixStride = 0;

        AscendC::Nd2NzParams dataCopyB1Params;
        dataCopyB1Params.ndNum = 1;
        dataCopyB1Params.nValue = k;
        dataCopyB1Params.dValue = n;
        dataCopyB1Params.srcNdMatrixStride = 0;
        dataCopyB1Params.srcDValue = n;
        dataCopyB1Params.dstNzC0Stride = k;
        dataCopyB1Params.dstNzNStride = 1;
        dataCopyB1Params.dstNzMatrixStride = 0;

        // AscendC::DataCopy GM->L1:ND->大N小z
        AscendC::DataCopy(a1Local, aGM, dataCopyA1Params);
        AscendC::DataCopy(b1Local, bGM, dataCopyB1Params);
        AscendC::DataCopy(deqLocal, deqTensorGM, deqTensorLen);
        
        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
        deqQueue.EnQue(deqLocal);
    }
    
    __aicore__ inline void SplitA()
    {
        AscendC::LocalTensor<a_T> a1Local = inQueueA1.DeQue<a_T>();
        AscendC::LocalTensor<a_T> a2Local = inQueueA2.AllocTensor<a_T>();

        AscendC::LoadData2dParams loadL0AParams;
        loadL0AParams.repeatTimes = mBlocks;
        loadL0AParams.srcStride = 1;
        loadL0AParams.dstGap = kBlocks - 1;
        loadL0AParams.ifTranspose = false;
        
        for (int i = 0; i < kBlocks; i++) {
            AscendC::LoadData(a2Local[i * AscendC::BLOCK_CUBE * (AscendC::ONE_BLK_SIZE / sizeof(a_T))], 
                             a1Local[i * m * (AscendC::ONE_BLK_SIZE / sizeof(a_T))], loadL0AParams);
        }

        inQueueA2.EnQue<a_T>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }
    
    __aicore__ inline void SplitB()
    {
        AscendC::LocalTensor<b_T> b1Local = inQueueB1.DeQue<b_T>();
        AscendC::LocalTensor<b_T> b2Local = inQueueB2.AllocTensor<b_T>();

        // load2d transpose L1->L0B
        AscendC::LoadData2dTransposeParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.srcStride = 1;
        loadDataParams.addrMode = 0;
        loadDataParams.repeatTimes = k * n / B8_SIZE;
        n_block = AscendC::ONE_BLK_SIZE;
        loadDataParams.dstGap = n / n_block - 1;
        loadDataParams.dstFracGap = 0;

        AscendC::LoadDataWithTranspose(b2Local, b1Local, loadDataParams);

        inQueueB1.FreeTensor(b1Local);
        inQueueB2.EnQue<b_T>(b2Local);
    }
    
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<a_T> a2Local = inQueueA2.DeQue<a_T>();
        AscendC::LocalTensor<b_T> b2Local = inQueueB2.DeQue<b_T>();
        AscendC::LocalTensor<dstCO1_T> c1Local = outQueueCO1.AllocTensor<dstCO1_T>();
        
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        
        AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams); // m*n
        outQueueCO1.EnQue<dstCO1_T>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }
    
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<dstCO1_T> c1Local = outQueueCO1.DeQue<dstCO1_T>();
        AscendC::LocalTensor<uint64_t> deqTensorLocal = deqQueue.DeQue<uint64_t>();
        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = m;
        fixpipeParams.dstStride = n;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 4;
        fixpipeParams.dstNdStride = m*n;
        fixpipeParams.quantPre = QuantMode_t::VDEQF16;
        
        AscendC::Fixpipe(cGM, c1Local, deqTensorLocal, fixpipeParams); // CO1到GM可以进行NZ到ND的转换
        outQueueCO1.FreeTensor(c1Local);
        deqQueue.FreeTensor(deqTensorLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::C1, 1> deqQueue;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;

    AscendC::GlobalTensor<a_T> aGM;
    AscendC::GlobalTensor<b_T> bGM;
    AscendC::GlobalTensor<c_T> cGM;
    AscendC::GlobalTensor<uint64_t> deqTensorGM;

    uint16_t m, k, n, n_mmad, startIndex, deqTensorLen;
    uint16_t B32_B16_SIZE = 16 * 16;
    uint16_t B8_SIZE = 32 * 32;
    uint8_t n_block = 16;
    bool L0Atranspose;
    uint8_t L0BtransposeMode;
    uint16_t aSize, bSize, cSize, b2Size, mBlocks, nBlocks, kBlocks;
};

#define KERNEL_MATMUL(c_type, a_type, b_type, dstCO1_type, mIn, kIn, nIn) \
extern "C" __global__ __aicore__ void cube_matmul_operator( \
    __gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *c, __gm__ uint8_t *deq) \
{ \
    if (g_coreType == AscendC::AIV) { \
        return; \
    } \
    KernelMatmul<c_type, a_type, b_type, dstCO1_type> op(mIn, kIn, nIn); \
    op.Init(a, b, c, deq); \
    op.Process(); \
}

KERNEL_MATMUL(half, int8_t, int8_t, int32_t, 32, 32, 32);
```

**示例结果：**

输入数据A矩阵、B矩阵、量化Tensor和输出数据C矩阵（具体数值略）
```
