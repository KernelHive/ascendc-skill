###### SoftmaxGradFront

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | ✓ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | ✓ |
| Atlas 200I/500 A2 推理产品 | ✓ |
| Atlas 推理系列产品AI Core | ✓ |
| Atlas 推理系列产品Vector Core | ✗ |
| Atlas 训练系列产品 | ✗ |

## 功能说明

将输入 tensor `[m0, m1, ...mt, n]`（t≥0）的非尾轴长度相乘的结果看作 m，则输入 tensor 的 shape 看作 `[m, n]`。对输入 tensor `[m, n]` 按行做 gradfront 反向计算，计算公式如下：

当输入 shape 为 ND 格式时，内部的 reduce 过程按 last 轴进行；当输入 shape 为 NZ 格式时，内部的 reduce 过程按照 last 轴和 first 轴进行，reduce 过程可以参考 SoftMax 中的图示说明。

为方便理解，通过 Python 脚本实现的方式，表达其计算公式如下，其中 dx、y 是源操作数（输入），d 为目的操作数（输出）。

```python
def softmax_grad_front(dx, y, is_fp16=False):
    dx = dx.astype(np.float32)
    y = y.astype(np.float32)
    
    d = (dx * y).sum(axis=-1, keepdims=True)  # [1024,1]
    if is_fp16:
        d = d.astype(np.float16)
    return d
```

## 实现原理

以 float 类型，ND 格式，shape 为 `[m, k]` 的输入 Tensor 为例，描述 SoftmaxGradFront 高阶 API 内部算法框图，如下图所示。

**图 15-52 SoftmaxGradFront 算法框图**

计算过程分为如下几步，均在 Vector 上进行：

1. **mul 步骤**：对输入 x 和 y 所有数据相乘，计算结果会保存到一个临时空间 temp 中；
2. **reducesum 步骤**：对 temp 中的数据 `[m, k]` 每一行数据求和得到 `[m, 1]`，计算结果保存到临时空间中；
3. **broadcast 步骤**：对 `[m, 1]` 做一个按 datablock 为单位的填充，比如 float 类型下，把 `[m, 1]` 扩展成 `[m, 8]`，并输出结果 z。

## 函数原型

### 接口框架申请临时空间

```cpp
template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGradFront(
    const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor,
    const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo = {}
)
```

### 通过 sharedTmpBuffer 入参传入临时空间

```cpp
template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGradFront(
    const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer,
    const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo = {}
)
```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持接口框架申请和开发者通过 sharedTmpBuffer 入参传入两种方式。

- **接口框架申请临时空间**：开发者无需申请，但是需要预留临时空间的大小。
- **通过 sharedTmpBuffer 入参传入**：使用该 tensor 作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理 sharedTmpBuffer 内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过 sharedTmpBuffer 传入的情况，开发者需要为 tensor 申请空间。临时空间大小 BufferSize 的获取方式如下：通过 SoftmaxGrad Tiling 接口中提供的 `GetSoftMaxGradMaxTmpSize`/`GetSoftMaxGradMinTmpSize` 接口获取所需最小和最大临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

## 参数说明

### 表 15-695 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 操作数的数据类型。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/float<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/float<br>Atlas 推理系列产品AI Core，支持的数据类型为：half/float<br>Atlas 200I/500 A2 推理产品，支持的数据类型为：half/float |
| isBasicBlock | srcTensor 和 gradTensor 的 shape 信息和 Tiling 切分策略满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。是否满足基本块的要求，可以采用如下两种方式之一判断：<br>• srcTensor 和 dstTensor 的 shape 信息 `[m,n]` 需要满足如下条件：<br>  – 尾轴长度 n 小于 2048 并且大于等于 256/sizeof(T)（即 half 场景下 n 最小为 128，float 场景下 n 最小为 64），同时 n 是 64 的倍数；<br>  – 非尾轴长度的乘积 m 为 8 的倍数。<br>• 在 Tiling 实现中，通过调用 `IsBasicBlockInSoftMax` 判断 Tiling 切分策略是否满足基本块的切分要求。<br>针对 Atlas 200I/500 A2 推理产品，该参数为预留参数，暂未启用，为后续的功能扩展做保留，保持默认值即可。 |
| isDataFormatNZ | 当前输入输出的数据格式是否为 NZ 格式，默认数据格式为 ND，即默认取值为 false。<br>针对 Atlas 200I/500 A2 推理产品，不支持配置为 NZ 格式。 |

### 表 15-696 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| dstTensor | 输出 | 目的操作数。<br>类型为 LocalTensor，支持的 TPosition 为 VECIN/VECCALC/VECOUT。<br>last 轴长度固定 32Byte 即一个 datablock 长度，并且该 datablock 中的所有数据为同一个值。比如 half 数据类型下，该 datablock 里的 16 个数均为相同的值，非 last 轴长度需要和 srcTensor 保持一致。 |
| gradTensor | 输入 | 源操作数。<br>类型为 LocalTensor，支持的 TPosition 为 VECIN/VECCALC/VECOUT。<br>last 轴长度需要 32Byte 对齐，gradTensor 的 shape 与 srcTensor 的 shape 一致。 |
| srcTensor | 输入 | 源操作数。<br>类型为 LocalTensor，支持的 TPosition 为 VECIN/VECCALC/VECOUT。<br>last 轴长度需要 32Byte 对齐，srcTensor 的 shape 与 gradTensor 的 shape 一致。 |
| sharedTmpBuffer | 输入 | 临时空间。<br>类型为 LocalTensor，支持的 TPosition 为 VECIN/VECCALC/VECOUT。<br>该操作数的数据类型固定 uint8_t。<br>用于接口内部复杂计算时存储中间变量，由开发者提供。<br>临时空间大小 BufferSize 的获取方式请参考 SoftmaxGrad Tiling 接口。 |
| tiling | 输入 | softmaxgradfront 计算所需 tiling 信息，Tiling 信息的获取请参考 SoftmaxGrad Tiling 接口。 |
| softmaxShapeInfo | 输入 | srcTensor 的 shape 信息。SoftMaxShapeInfo 类型，具体定义如下：<br>```cpp<br>struct SoftMaxShapeInfo {<br>    uint32_t srcM;    // 非尾轴乘积长度<br>    uint32_t srcK;    // 尾轴长度，必须 32Byte 对齐<br>    uint32_t oriSrcM; // 原始非尾轴乘积长度<br>    uint32_t oriSrcK; // 原始尾轴长度<br>};<br>```<br>需要注意，当输入输出的数据格式为 NZ 格式时，尾轴长度为 reduce 轴长度即图 15-47 中的 W0*W1，非尾轴为 H0*H1。 |

## 返回值说明

无

## 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束。
- 不支持 sharedTmpBuffer 与源操作数和目的操作数地址重叠。
- 当参数 softmaxShapeInfo 中 `srcM != oriSrcM` 或者 `srcK != oriSrcK` 时，开发者需要对 GM 上的原始输入 `(oriSrcM, oriSrcK)` 在 M 或 K 方向补齐数据到 `(srcM, srcK)`，补齐的数据会参与部分运算，在输入输出复用的场景下，API 的计算结果会覆盖 srcTensor 中补齐的原始数据，在输入输出不复用的场景下，API 的计算结果会覆盖 dstTensor 中对应 srcTensor 补齐位置的数据。

## 调用示例

本样例输入 srcTensor 的 Shape 大小为 `[128,64]`，输入 gradtensor 的 Shape 大小为 `[128,64]`，输出 dstTensor 的 Shape 大小为 `[128,16]`，数据类型均为 half，输入输出的数据排布格式为 ND，不使能基本块。更多算子样例请参考 softmaxgradfront 算子样例。

```cpp
#include "kernel_operator.h"

template <typename T> class KernelSoftmaxGrad {
public:
    __aicore__ inline KernelSoftmaxGrad() {}
    __aicore__ inline void Init(__gm__ uint8_t* src1Gm, __gm__ uint8_t* src2Gm, __gm__ uint8_t* dstGm, const SoftMaxTiling& tilingData)
    {
        elementNumPerBlk = 32 / sizeof(T);
        src1Global.SetGlobalBuffer((__gm__ T*)src1Gm);
        src2Global.SetGlobalBuffer((__gm__ T*)src2Gm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrc1, 1, height * width * sizeof(T));
        pipe.InitBuffer(inQueueSrc2, 1, height * width * sizeof(T));
        pipe.InitBuffer(outQueueDst, 1, height * elementNumPerBlk * sizeof(T));
        tiling = tilingData;
    }
    
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<T> srcLocal1 = inQueueSrc1.AllocTensor<T>();
        AscendC::LocalTensor<T> srcLocal2 = inQueueSrc2.AllocTensor<T>();
        AscendC::DataCopy(srcLocal1, src1Global, height * width);
        AscendC::DataCopy(srcLocal2, src2Global, height * width);
        inQueueSrc1.EnQue(srcLocal1);
        inQueueSrc2.EnQue(srcLocal2);
    }
    
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> srcLocal1 = inQueueSrc1.DeQue<T>();
        AscendC::LocalTensor<T> srcLocal2 = inQueueSrc2.DeQue<T>();
        AscendC::LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();
        AscendC::SoftMaxShapeInfo srcShape = { height, width, height, width };
        AscendC::SoftmaxGradFront<T>(dstLocal, srcLocal2, srcLocal1, tiling, srcShape);
        outQueueDst.EnQue<T>(dstLocal);
        inQueueSrc1.FreeTensor(srcLocal1);
        inQueueSrc2.FreeTensor(srcLocal2);
    }
    
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dstLocal = outQueueDst.DeQue<T>();
        AscendC::DataCopy(dstGlobal, dstLocal, height * elementNumPerBlk);
        outQueueDst.FreeTensor(dstLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc1;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc2;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<T> src1Global, src2Global, dstGlobal;
    uint32_t elementNumPerBlk = 0;
    uint32_t width = 64;
    uint32_t height = 128;
    SoftMaxTiling tiling;
};

extern "C" __global__ __aicore__ void softmax_grad_kernel_half(__gm__ uint8_t* src1Gm, __gm__ uint8_t* src2Gm, __gm__ uint8_t* dstGm, __gm__ uint8_t* tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelSoftmaxGrad<half> op;
    op.Init(src1Gm, src2Gm, dstGm, tilingData.softmaxTilingData);
    op.Process();
}
```
