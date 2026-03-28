##### DeepNorm Tiling

## 返回值说明

无

## 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束
- srcLocal和dstLocal的Tensor空间可以复用
- 仅支持输入shape为ND格式
- 输入数据不满足对齐要求时，开发者需要进行补齐，补齐的数据应设置为0，防止出现异常值从而影响网络计算

## 调用示例

```cpp
#include "kernel_operator.h"

template <typename dataType, bool isReuseSrc = false, bool isBasicBlock = false>
class KernelDeepNorm {
public:
    __aicore__ inline KernelDeepNorm()
    {}
    
    __aicore__ inline void Init(GM_ADDR inputGm, GM_ADDR inputGxGm, GM_ADDR betaGm, GM_ADDR gammaGm, GM_ADDR outputGm,
                               GM_ADDR outputMeanGm, GM_ADDR outputVarianceGm, const DeepNormCustomTiling &customTiling)
    {
        this->tiling = customTiling.tiling; // DeepNormTiling
        alpha = customTiling.alpha;
        epsilon = customTiling.epsilon;
        const uint32_t bLength = tiling.bLength;
        const uint32_t sLength = tiling.sLength;
        hLength = tiling.hLength;
        bshLength = bLength * sLength * hLength;
        bsLength = bLength * sLength;
        
        inputXGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(inputGm), bshLength);
        inputGxGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(inputGxGm), bshLength);
        betaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(betaGm), hLength);
        gammaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gammaGm), hLength);
        outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(outputGm), bshLength);
        outputMeanGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(outputMeanGm), bsLength);
        outputVarianceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(outputVarianceGm), bsLength);
        
        constexpr uint32_t typeSize = sizeof(dataType);
        pipe.InitBuffer(inQueueX, 1, bshLength * typeSize);
        pipe.InitBuffer(inQueueGx, 1, bshLength * typeSize);
        pipe.InitBuffer(inQueueBeta, 1, hLength * typeSize);
        pipe.InitBuffer(inQueueGamma, 1, hLength * typeSize);
        pipe.InitBuffer(outQueue, 1, bshLength * typeSize);
        pipe.InitBuffer(outMeanQueue, 1, bsLength * typeSize);
        pipe.InitBuffer(outVarianceQueue, 1, bsLength * typeSize);
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
        AscendC::LocalTensor<dataType> inputXLocal = inQueueX.AllocTensor<dataType>();
        AscendC::LocalTensor<dataType> inputGxLocal = inQueueGx.AllocTensor<dataType>();
        AscendC::LocalTensor<dataType> betaLocal = inQueueBeta.AllocTensor<dataType>();
        AscendC::LocalTensor<dataType> gammaLocal = inQueueGamma.AllocTensor<dataType>();
        
        AscendC::DataCopy(inputXLocal, inputXGlobal, bshLength);
        AscendC::DataCopy(inputGxLocal, inputGxGlobal, bshLength);
        AscendC::DataCopy(betaLocal, betaGlobal, hLength);
        AscendC::DataCopy(gammaLocal, gammaGlobal, hLength);
        
        inQueueX.EnQue(inputXLocal);
        inQueueGx.EnQue(inputGxLocal);
        inQueueBeta.EnQue(betaLocal);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<dataType> inputXLocal = inQueueX.DeQue<dataType>();
        AscendC::LocalTensor<dataType> inputGxLocal = inQueueGx.DeQue<dataType>();
        AscendC::LocalTensor<dataType> betaLocal = inQueueBeta.DeQue<dataType>();
        AscendC::LocalTensor<dataType> gammaLocal = inQueueGamma.DeQue<dataType>();
        AscendC::LocalTensor<dataType> outputLocal = outQueue.AllocTensor<dataType>();
        AscendC::LocalTensor<dataType> outputMeanLocal = outMeanQueue.AllocTensor<dataType>();
        AscendC::LocalTensor<dataType> outputVarianceLocal = outVarianceQueue.AllocTensor<dataType>();

        AscendC::DeepNorm<dataType, isReuseSrc, isBasicBlock>(outputLocal,
                                                             outputMeanLocal,
                                                             outputVarianceLocal,
                                                             inputXLocal,
                                                             inputGxLocal,
                                                             betaLocal,
                                                             gammaLocal,
                                                             alpha,
                                                             epsilon,
                                                             tiling);

        inQueueX.FreeTensor(inputXLocal);
        inQueueGx.FreeTensor(inputGxLocal);
        inQueueBeta.FreeTensor(betaLocal);
        inQueueGamma.FreeTensor(gammaLocal);
        outQueue.EnQue(outputLocal);
        outMeanQueue.EnQue(outputMeanLocal);
        outVarianceQueue.EnQue(outputVarianceLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<dataType> outputLocal = outQueue.DeQue<dataType>();
        AscendC::LocalTensor<dataType> outputMeanLocal = outMeanQueue.DeQue<dataType>();
        AscendC::LocalTensor<dataType> outputVarianceLocal = outVarianceQueue.DeQue<dataType>();
        
        AscendC::DataCopy(outputGlobal, outputLocal, bshLength);
        AscendC::DataCopy(outputMeanGlobal, outputMeanLocal, bsLength);
        AscendC::DataCopy(outputVarianceGlobal, outputVarianceLocal, bsLength);
        
        outQueue.FreeTensor(outputLocal);
        outMeanQueue.FreeTensor(outputMeanLocal);
        outVarianceQueue.FreeTensor(outputVarianceLocal);
    }

private:
    AscendC::GlobalTensor<dataType> inputXGlobal;
    AscendC::GlobalTensor<dataType> inputGxGlobal;
    AscendC::GlobalTensor<dataType> betaGlobal;
    AscendC::GlobalTensor<dataType> gammaGlobal;
    AscendC::GlobalTensor<dataType> outputGlobal;
    AscendC::GlobalTensor<dataType> outputMeanGlobal;
    AscendC::GlobalTensor<dataType> outputVarianceGlobal;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueGx;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueBeta;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueGamma;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outMeanQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outVarianceQueue;

    DeepNormTiling tiling;
    uint32_t bshLength;
    uint32_t bsLength;
    uint32_t hLength;
    dataType alpha;
    dataType epsilon;
};

template <typename dataType, bool isReuseSrc = false, bool isBasicBlock = false>
__aicore__ inline void kernel_deepnorm_operator(GM_ADDR inputGm, GM_ADDR inputGxGm, GM_ADDR betaGm, GM_ADDR gammaGm,
                                               GM_ADDR outputGm, GM_ADDR outputMeanGm, GM_ADDR outputVarianceGm, GM_ADDR customTiling)
{
    GET_TILING_DATA(tilingData, customTiling)
    KernelDeepNorm<dataType, isReuseSrc, isBasicBlock> op;
    op.Init(inputGm, inputGxGm, betaGm, gammaGm, outputGm, outputMeanGm, outputVarianceGm, tilingData);
    op.Process();
}
```

## 功能说明

Ascend C提供DeepNorm Tiling API，方便用户获取DeepNorm kernel计算时所需的Tiling参数。

获取Tiling参数主要分为如下两步：

1. **通过GetDeepNormMaxMinTmpSize获取DeepNorm接口计算所需最大和最小临时空间大小**
   
   kernel侧DeepNorm接口的计算需要开发者预留/申请临时空间，GetDeepNormMaxMinTmpSize用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。
   
   - 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小
   - 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请

2. **通过GetDeepNormTilingInfo获取DeepNorm kernel侧接口所需tiling参数**
   
   DeepNormTiling结构体的定义如下，开发者无需关注该tiling结构的具体信息，只需要传递到kernel侧，传入DeepNorm高阶API接口，直接进行使用即可。

```cpp
struct DeepNormTiling {
    uint32_t bLength = 0;
    uint32_t sLength = 0;
    uint32_t hLength = 0;
    uint32_t originalHLength = 0;
    uint32_t inputXSize = 0;
    uint32_t meanVarSize = 0;
    uint32_t numberOfTmpBuf = 0;
    uint32_t meanTmpTensorPos = 0;
    uint32_t meanTmpTensorSize = 0;
    uint32_t varianceTmpTensorPos = 0;
    uint32_t varianceTmpTensorSize = 0;
    uint32_t tmpBufSize = 0;
    uint32_t oneTmpSize = 0;
    uint32_t firstTmpStartPos = 0;
    uint32_t secondTmpStartPos = 0;
    uint32_t thirdTmpStartPos = 0;
    uint32_t loopRound = 0;
    uint32_t inputRoundSize = 0;
    uint32_t inputTailSize = 0;
    uint32_t inputTailPos = 0;
    uint32_t meanVarRoundSize = 0;
    uint32_t meanVarTailSize = 0;
    uint32_t meanVarTailPos = 0;
    uint32_t bshCurLength = 0;
    uint32_t bsCurLength = 0;
    float lastDimValueBack = 0;
};
```

## 函数原型

```cpp
bool GetDeepNormMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, const bool isBasicBlock, uint32_t& maxValue, uint32_t& minValue)

bool GetDeepNormTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, const bool isBasicBlock, optiling::DeepNormTiling& tiling)
```

## 参数说明

### GetDeepNormMaxMinTmpSize 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| srcShape | 输入 | 输入的shape信息 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2 |
| isReuseSource | 输入 | 是否复用源操作数输入的空间，与DeepNorm接口一致 |
| isBasicBlock | 输入 | srcShape是否符合基本块定义：尾轴H的长度为64的倍数（不超过2040），B*S为8的倍数 |
| maxValue | 输出 | DeepNorm接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。<br>**说明：** maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小 |
| minValue | 输出 | DeepNorm接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间 |

### GetDeepNormTilingInfo 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| srcShape | 输入 | 输入的shape信息[B, S, H] |
| originSrcShape | 输入 | 32B对齐前的输入shape信息 [B, S, originH]。originH的长度应该在 (0, H]的范围内。如果isBasicBlock置为true，originH必须与H一致 |
| stackBufferSize | 输入 | 临时空间的buffer大小，单位为Byte。通过GetDeepNormMaxMinTmpSize获取最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为stackBufferByteSize传入 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2 |
| isReuseSource | 输入 | 是否复用源操作数输入的空间，与DeepNorm接口一致 |
| isBasicBlock | 输入 | srcShape是否符合基本块定义：尾轴H的长度为64的倍数（不超过2040），B*S为8的倍数 |
| tiling | 输出 | DeepNorm计算所需Tiling信息 |

## 返回值说明

- **GetDeepNormMaxMinTmpSize** 返回值为true/false，true表示成功拿到DeepNorm接口内部计算需要的最大和最小临时空间大小；false表示获取失败
- **GetDeepNormTilingInfo** 返回类型为true/false，true表示成功拿到DeepNorm的Tiling各项参数值；false表示获取失败

## 约束说明

无

## 调用示例

### 步骤1：将DeepNorm Tiling结构体参数增加至TilingData结构体

```cpp
BEGIN_TILING_DATA_DEF(TilingData) // 注册一个tiling的类，以tiling的名字作为入参
    TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 添加tiling字段，总计算数据量
    TILING_DATA_FIELD_DEF(uint32_t, tileNum); // 添加tiling字段，每个核上总计算数据分块个数
    ... // 添加其他tiling字段
    TILING_DATA_FIELD_DEF_STRUCT(DeepNormTiling, deepnormTilingData); // 将DeepNormTiling结构体参数增加至TilingData结构体
END_TILING_DATA_DEF;
```

### 步骤2：Tiling实现函数

```cpp
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    // 设置其他Tiling参数
    ...
    
    std::vector<int64_t> shapeVec = {2, 16, 64};
    std::vector<int64_t> oriShapeVec = {2, 16, 64};
    ge::Shape srcShape(shapeVec);
    ge::Shape originSrcShape(oriShapeVec);

    // 本样例中仅作为样例说明，通过GetDeepNormMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
    uint32_t minValue = 0;
    uint32_t maxValue = 0;
    AscendC::GetDeepNormMaxMinTmpSize(srcShape, sizeof(half), isReuseSrc, isBasicBlock, maxValue, minValue);

    // 获取DeepNorm Tiling参数
    AscendC::GetDeepNormTilingInfo(srcShape, originSrcShape, minValue, sizeof(half), isReuseSrc, isBasicBlock, tiling.deepnormTilingData);

    ... // 其他逻辑
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

### 步骤3：Kernel侧调用

```cpp
extern "C" __global__ __aicore__ void deepnorm_custom(GM_ADDR inputX, GM_ADDR inputGx, GM_ADDR beta, GM_ADDR gamma, GM_ADDR output, GM_ADDR outputMean, GM_ADDR outputVariance, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelDeepNorm op;
    op.Init(inputX, inputGx, beta, gamma, output, outputMean, outputVariance, tilingData.totalLength, tilingData.tileNum, tilingData.deepnormTilingData);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
```
