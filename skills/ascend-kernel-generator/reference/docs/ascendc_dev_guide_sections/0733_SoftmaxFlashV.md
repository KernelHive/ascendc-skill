###### SoftmaxFlashV

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

SoftmaxFlash增强版本，对应Softmax PASA算法。将输入tensor `[m0, m1, ..., mt, n]`（t大于或等于0）的非尾轴长度 `m0, m1, ..., mt` 相乘的结果看作 `m`，则输入tensor的shape看作 `[m, n]`。对输入tensor x的尾轴进行切分，分块个数为 `splitMeanCnt`，切分后的tensor为 `x_cnti`。按如下公式进行计算，其中 `x`、`inmax`、`insum`、`inmean` 为输入，`M`、`S`、`E` 均为输出。

- **update为false**：公式未提供
- **update为true**：公式未提供

本接口当前只支持ND格式的输入，内部的reduce过程按last轴处理。

为方便理解，通过Python伪代码实现的方式，表达其计算公式如下。其中，`repeatSize` 为64，`elementNumPerBlk/BlkcntPerRepeat` 为8，`splitMeanCnt` 为8，`src`、`inmean`、`inmax`、`insum`、`update` 为输入，`dst`、`x_mean`、`x_sum`、`x_max`、`exp_max` 为输出。

```python
def softmax_flash_3(src, height, width, loopCnt, alpha, baseK, inmax=None, insum=None, inmean=None, update=False):
    scalar = alpha / (1 - alpha)
    #(m,n)->(m,64)
    tmpbuffer0 = BlockReduceSum(repeatSize, repeatSize, elementNumPerBlk)
    remain = int(width / repeatSize - BlkcntPerRepeat)
    tmpbuffer0 = Add(tmpbuffer0, src, remain, repeatSize * elementNumPerBlk, width)
    #(m,64)->(m,8)
    tmpbuffer0 = BlockReduceSum(1, elementNumPerBlk, elementNumPerBlk)
    #width = baseK * splitMeanCnt
    rowMeanLocal = tmpbuffer0 / baseK
    rowMeanGlobal = np.mean(src, axis=(-1), keepdims=True)
    rowMeanGlobalTmp = (rowMeanGlobal - rowMeanLocal) * scalar
    src = src - rowMeanGlobalTmp

    if update == False:
        x_mean = rowMeanGlobal
        maxTmp = np.max(src, axis=-1, keepdims=True)
        shiftCurr = (rowMeanGlobal - x_mean) * scalar
        x_max = shiftCurr + maxTmp
        maxTmp = x_max - shiftCurr
        x_sub = src - maxTmp
        
        dst = np.exp(x_sub)
        x_sum = np.sum(dst, axis=-1, keepdims=True)
        exp_max = None
        return dst, x_max, x_sum, x_mean, exp_max
    else:
        x_mean = (rowMeanGlobal + inmean * (loopCnt - 1)) / loopCnt
        maxTmp = np.max(src, axis=-1, keepdims=True)
        shiftCurr = (rowMeanGlobal - x_mean) * scalar
        shiftPrev = (inmean - x_mean) * scalar
        x_max = shiftCurr + maxTmp
        maxTmp = shiftPrev + inmax
        x_max = np.max(np.concatenate((x_max, maxTmp), axis=(-1)), axis=(-1), keepdims=True)
        maxTmp = x_max - shiftCurr
        x_sub = src - maxTmp
        dst = np.exp(x_sub)
        exp_max = np.exp(inmax - x_max + shiftPrev)
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)
        x_sum = exp_max * insum + x_sum
        return x_exp, x_max, x_sum, x_mean, exp_max
```

## 函数原型

### 接口框架申请临时空间

```cpp
template <typename T, typename U, bool isUpdate = false, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftmaxFlashV3(const LocalTensor<T>& dstTensor, const LocalTensor<U>& meanTensor, const LocalTensor<U>& expSumTensor, const LocalTensor<U>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<U>& inMeanTensor, const LocalTensor<U>& inExpSumTensor, const LocalTensor<U>& inMaxTensor, const SoftMaxTiling& tiling, const SoftMaxParams& params)
```

### 通过sharedTmpBuffer入参传入临时空间

```cpp
template <typename T, typename U, bool isUpdate = false, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftmaxFlashV3(const LocalTensor<T>& dstTensor, const LocalTensor<U>& meanTensor, const LocalTensor<U>& expSumTensor, const LocalTensor<U>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<U>& inMeanTensor, const LocalTensor<U>& inExpSumTensor, const LocalTensor<U>& inMaxTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxParams& params)
```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持接口框架申请和开发者通过sharedTmpBuffer入参传入两种方式。

- **接口框架申请临时空间**：开发者无需申请，但是需要预留临时空间的大小。
- **通过sharedTmpBuffer入参传入**：使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过SoftmaxFlashV3 Tiling接口中提供的GetSoftMaxFlashV3MaxMinTmpSize接口获取所需最小和最大临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

## 参数说明

### 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 输入srcTensor及输出dstTensor、expMaxTensor操作数的数据类型。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half |
| U | 输入inMeanTensor、inExpSumTensor、inMaxTensor及输出meanTensor、expSumTensor、maxTensor操作数的数据类型。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：float<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：float |
| isUpdate | 是否使能update为true的计算。 |
| isReuseSource | 该参数预留，传入默认值false即可。 |
| isBasicBlock | 该参数预留，传入默认值false即可。 |
| isDataFormatNZ | 该参数预留，传入默认值false即可。 |
| config | 该参数预留，传入默认值SOFTMAX_DEFAULT_CFG即可。 |

### 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| dstTensor | 输出 | 目的操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>dstTensor的shape和源操作数srcTensor一致。 |
| meanTensor | 输出 | 目的操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>用于保存softmax计算过程中平均值的结果。<br>• meanTensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值。比如float数据类型下，该datablock中的8个数均为相同的reducesum求平均后的值。<br>• 非last轴的长度与dstTensor保持一致。 |
| expSumTensor | 输出 | 目的操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>用于保存softmax计算过程中reducesum的结果。<br>• expSumTensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值。比如float数据类型下，该datablock中的8个数均为相同的reducesum的值。<br>• 非last轴的长度与dstTensor保持一致。 |
| maxTensor | 输出 | 目的操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>用于保存softmax计算过程中reducemax的结果。<br>• maxTensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值。比如float数据类型下，该datablock中的8个数均为相同的reducemax的值。<br>• 非last轴的长度与dstTensor保持一致。 |
| srcTensor | 输入 | 源操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>last轴长度需要32Byte对齐。 |
| expMaxTensor | 输出 | 目的操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>• expMaxTensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值。比如half数据类型下，该datablock中的16个数均为相同的值。<br>• 非last轴的长度需要与dstTensor保持一致。 |
| inMeanTensor | 输入 | 源操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>softmax计算所需要的mean值。<br>• inMeanTensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值。比如float数据类型下，该datablock中的8个数均为相同的值。<br>• 非last轴的长度需要与dstTensor保持一致。 |
| inExpSumTensor | 输入 | 源操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>softmax计算所需要的sum值。<br>• inExpSumTensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值。比如float数据类型下，该datablock中的8个数均为相同的值。<br>• 非last轴的长度需要与dstTensor保持一致。 |
| inMaxTensor | 输入 | 源操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>softmax计算所需要的max值。<br>• inMaxTensor的last轴长度固定为32Byte，即一个datablock长度。该datablock中的所有数据为同一个值。比如float数据类型下，该datablock中的8个数均为相同的值。<br>• 非last轴的长度需要与dstTensor保持一致。 |
| sharedTmpBuffer | 输入 | 临时空间。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>该操作数的数据类型固定uint8_t。<br>接口内部复杂计算时用于存储中间变量，由开发者提供。<br>临时空间大小BufferSize的获取方式请参考SoftmaxFlashV3 Tiling接口。 |
| tiling | 输入 | SoftmaxFlashV3接口计算所需Tiling信息，Tiling信息的获取请参考SoftmaxFlashV3 Tiling接口。 |
| params | 输入 | srcTensor的shape信息和计算相关参数。SoftMaxParams类型，具体定义如下：<br>```cpp<br>struct SoftMaxParams {<br>    uint32_t srcM;          // 非尾轴长度的乘积<br>    uint32_t srcK;          // 尾轴长度，必须32Byte对齐<br>    uint32_t oriSrcM;       // 原始非尾轴长度的乘积<br>    uint32_t oriSrcK;       // 原始尾轴长度<br>    uint32_t loopCnt;       // update为true时，公式中的循环次数loopCnt，该参数大于等于1<br>    uint32_t splitMeanCnt;  // 公式中计算每一行平均值时的分块个数，当前该参数仅支持取值为8<br>    float alpha;            // 公式中的计算参数，推荐取值0.9375、0.96889、0.984497<br>};<br>```<br>注意，当前本接口不支持非对齐场景，因此参数srcM与oriSrcM相等，参数srcK与oriSrcK相等。 |

## 返回值说明

无

## 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束。
- 对于输入srcTensor需要满足：尾轴长度n大于等于512，同时n是64的倍数；非尾轴长度的乘积m为8的倍数。
- srcTensor和dstTensor的Tensor的空间可以复用，meanTensor和inMeanTensor的空间可以复用，maxTensor和inMaxTensor的空间可以复用，expSumTensor和inExpSumTensor的空间可以复用。
- meanTensor、expSumTensor、maxTensor、expMaxTensor、inMeanTensor、inExpSumTensor、inMaxTensor的Tensor空间，last轴长度必须是32字节。
- 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

## 调用示例

本样例中输入srcTensor和输出dstTensor的shape大小为[8, 1024]，输入inMeanTensor、inExpSumTensor、inMaxTensor的shape大小为[8, 8]，数据类型为float；输出expMaxTensor的shape大小为[8, 16]，数据类型为half；输入和输出的数据排布格式为ND，srcTensor和dstTensor空间不复用，模板参数isUpdate为true。

```cpp
#include "kernel_operator.h"

template <typename T, typename U>
class KernelSoftmaxFlashV3 {
public:
    __aicore__ inline KernelSoftmaxFlashV3() {}
    __aicore__ inline void Init(__gm__ uint8_t *srcGm, __gm__ uint8_t *inMaxGm, __gm__ uint8_t *inSumGm,
                                __gm__ uint8_t *inMeanGm, __gm__ uint8_t *dstGm, const SoftMaxTiling &tilingData)
    {
        srcGlobal.SetGlobalBuffer((__gm__ T *)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T *)dstGm);
        maxGlobal.SetGlobalBuffer((__gm__ U *)inMaxGm);
        sumGlobal.SetGlobalBuffer((__gm__ U *)inSumGm);
        meanGlobal.SetGlobalBuffer((__gm__ U *)inMeanGm);
        pipe.InitBuffer(inQueueSrc, 1, height * width * sizeof(T));
        elementNumPerBlk1 = 32 / sizeof(U);
        pipe.InitBuffer(maxQueue, 1, height * elementNumPerBlk1 * sizeof(U));
        pipe.InitBuffer(sumQueue, 1, height * elementNumPerBlk1 * sizeof(U));
        pipe.InitBuffer(meanQueue, 1, height * elementNumPerBlk1 * sizeof(U));
        elementNumPerBlk2 = 32 / sizeof(T);
        pipe.InitBuffer(expMaxQueue, 1, height * elementNumPerBlk2 * sizeof(T));
        pipe.InitBuffer(outQueueDst, 1, height * width * sizeof(T));
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
        AscendC::LocalTensor<T> srcLocal = inQueueSrc.AllocTensor<T>();
        AscendC::LocalTensor<U> insumLocal = sumQueue.AllocTensor<U>();
        AscendC::LocalTensor<U> inmaxLocal = maxQueue.AllocTensor<U>();
        AscendC::LocalTensor<U> inmeanLocal = meanQueue.AllocTensor<U>();
        AscendC::DataCopy(srcLocal, srcGlobal, height * width);
        
        AscendC::DataCopy(insumLocal, sumGlobal, height * elementNumPerBlk1);
        AscendC::DataCopy(inmaxLocal, maxGlobal, height * elementNumPerBlk1);
        AscendC::DataCopy(inmeanLocal, meanGlobal, height * elementNumPerBlk1);
        inQueueSrc.EnQue(srcLocal);
        sumQueue.EnQue(insumLocal);
        maxQueue.EnQue(inmaxLocal);
        meanQueue.EnQue(inmeanLocal);
    }
    
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
        AscendC::LocalTensor<U> insumLocal = sumQueue.DeQue<U>();
        AscendC::LocalTensor<U> inmaxLocal = maxQueue.DeQue<U>();
        AscendC::LocalTensor<U> inmeanLocal = meanQueue.DeQue<U>();
        AscendC::LocalTensor<T> expMaxTensor = expMaxQueue.AllocTensor<T>();
        AscendC::LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();
        AscendC::SoftMaxParams params = {height, width, height, width, loopCnt, splitMeanCnt, alpha};
        AscendC::SoftmaxFlashV3<T, U, true>(dstLocal, inmeanLocal, insumLocal, inmaxLocal, srcLocal,
                                           expMaxTensor, inmeanLocal, insumLocal, inmaxLocal, tiling, params);

        outQueueDst.EnQue<T>(dstLocal);
        maxQueue.FreeTensor(inmaxLocal);
        sumQueue.FreeTensor(insumLocal);
        meanQueue.FreeTensor(inmeanLocal);
        inQueueSrc.FreeTensor(srcLocal);
    }
    
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dstLocal = outQueueDst.DeQue<T>();
        AscendC::DataCopy(dstGlobal, dstLocal, height * width);
        outQueueDst.FreeTensor(dstLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> meanQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> maxQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> sumQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> expMaxQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<T> srcGlobal, dstGlobal;
    AscendC::GlobalTensor<U> meanGlobal, maxGlobal, sumGlobal;
    uint32_t elementNumPerBlk1 = 0;
    uint32_t elementNumPerBlk2 = 0;
    uint32_t width = 1024;
    uint32_t height = 8;
    uint32_t loopCnt = 2;
    uint32_t splitMeanCnt = 8;
    float alpha = 0.9375;
    SoftMaxTiling tiling;
};

extern "C" __global__ __aicore__ void softmax_flashv3_kernel(__gm__ uint8_t *srcGm,
                                                           __gm__ uint8_t *inMaxGm, __gm__ uint8_t *inSumGm, 
                                                           __gm__ uint8_t *inMeanGm, __gm__ uint8_t *dstGm, 
                                                           __gm__ uint8_t *tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelSoftmaxFlashV3<half, float> op;
    op.Init(srcGm, inMaxGm, inSumGm, inMeanGm, dstGm, tilingData.softmaxTilingData);
    op.Process();
}
```
