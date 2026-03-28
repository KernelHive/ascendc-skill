##### RmsNorm Tiling

## 返回值说明
无

## 约束说明
- `srcLocal` 和 `dstLocal` 的 Tensor 空间可以复用
- 当前仅支持 ND 格式的输入，不支持其他格式
- 操作数地址对齐要求请参见通用地址对齐约束

## 调用示例

```cpp
#include "kernel_operator.h"

inline __aicore__ uint32_t AlignToBlock(const uint32_t inputValue, const uint32_t typeSize)
{
    constexpr uint32_t ONE_BLK_SIZE = 32;
    uint32_t alignUnit = ONE_BLK_SIZE / typeSize;
    return (inputValue + alignUnit - 1) / alignUnit * alignUnit;
}

template <typename dataType, bool isBasicBlock = false>
class KernelRmsNorm {
public:
    __aicore__ inline KernelRmsNorm()
    {}
    
    __aicore__ inline void Init(
        GM_ADDR inputGm, 
        GM_ADDR gammaGm, 
        GM_ADDR outputGm, 
        const RmsNormCustomTiling &customTiling)
    {
        tiling = customTiling.tiling;
        const uint32_t bLength = tiling.bLength;
        const uint32_t sLength = tiling.sLength;
        hLength = tiling.hLength;
        bshLength = bLength * sLength * hLength;
        constexpr uint32_t typeSize = sizeof(dataType);
        const uint32_t bsLength = AlignToBlock(bLength * sLength, typeSize);
        const uint32_t tmpBufferSize = bshLength * 2 + bsLength;
        epsilon = customTiling.epsilon;
        
        inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(inputGm), bshLength);
        gammaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gammaGm), hLength);
        outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(outputGm), bshLength);

        pipe.InitBuffer(inQueue, 1, bshLength * typeSize);
        pipe.InitBuffer(inQueueGamma, 1, hLength * typeSize);
        pipe.InitBuffer(outQueue, 1, bshLength * typeSize);
        pipe.InitBuffer(tmpQueue, 1, tmpBufferSize);
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
        AscendC::LocalTensor<dataType> inputLocal = inQueue.AllocTensor<dataType>();
        AscendC::DataCopy(inputLocal, inputGlobal, bshLength);
        inQueue.EnQue(inputLocal);
        
        AscendC::LocalTensor<dataType> gammaLocal = inQueueGamma.AllocTensor<dataType>();
        AscendC::DataCopy(gammaLocal, gammaGlobal, hLength);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<dataType> inputLocal = inQueue.DeQue<dataType>();
        AscendC::LocalTensor<dataType> gammaLocal = inQueueGamma.DeQue<dataType>();
        AscendC::LocalTensor<dataType> outputLocal = outQueue.AllocTensor<dataType>();
        AscendC::LocalTensor<uint8_t> stackBuffer = tmpQueue.AllocTensor<uint8_t>();
        
        AscendC::RmsNorm<dataType, isBasicBlock>(outputLocal, inputLocal, gammaLocal, stackBuffer, epsilon, tiling);
        
        inQueue.FreeTensor(inputLocal);
        inQueueGamma.FreeTensor(gammaLocal);
        tmpQueue.FreeTensor(stackBuffer);
        outQueue.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<dataType> outputLocal = outQueue.DeQue<dataType>();
        AscendC::DataCopy(outputGlobal, outputLocal, bshLength);
        outQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::GlobalTensor<dataType> inputGlobal;
    AscendC::GlobalTensor<dataType> gammaGlobal;
    AscendC::GlobalTensor<dataType> outputGlobal;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueGamma;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpQueue;
    RmsNormTiling tiling;
    uint32_t hLength;
    dataType epsilon;
    uint32_t bshLength;
};

template <typename dataType, bool isBasicBlock = false>
__aicore__ inline void kernel_rmsnorm_operator(GM_ADDR inputGm, GM_ADDR gammaGm, GM_ADDR outputGm, GM_ADDR tiling)
{
    GET_TILING_DATA(customTilingData, tiling)
    KernelRmsNorm<dataType, isBasicBlock> op;
    op.Init(inputGm, gammaGm, outputGm, customTilingData);
    op.Process();
}
```

## 功能说明
Ascend C 提供 RmsNorm Tiling API，方便用户获取 RmsNorm kernel 计算时所需的 Tiling 参数。

获取 Tiling 参数主要分为如下两步：

1. **通过 `GetRmsNormMaxMinTmpSize` 获取 RmsNorm 接口计算所需最大和最小临时空间大小**
   - kernel 侧 RmsNorm 接口的计算需要开发者预留/申请临时空间
   - `GetRmsNormMaxMinTmpSize` 用于在 host 侧获取预留/申请的最大最小临时空间大小
   - 开发者基于此范围选择合适的空间大小作为 Tiling 参数传递到 kernel 侧使用
   - 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小
   - 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel 侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请

2. **通过 `GetRmsNormTilingInfo` 获取 RmsNorm kernel 侧接口所需 tiling 参数**
   - RmsNorm Tiling 结构体的定义如下，开发者无需关注该 tiling 结构的具体信息，只需要传递到 kernel 侧，传入 RmsNorm 高阶 API 接口，直接进行使用即可

```cpp
struct RmsNormTiling {
    uint32_t bLength = 0;
    uint32_t sLength = 0;
    uint32_t hLength = 0;
    uint32_t originalHLength = 0;
    float reciprocalOfHLength = 0;
    uint32_t mainBshLength = 0;
    uint32_t mainBsLength = 0;
    uint32_t mainBsLengthAlign = 0;
    uint32_t loopRound = 0;
    uint32_t inputTailPos = 0;
    uint32_t tailBshLength = 0;
    uint32_t tailBsLength = 0;
};
```

## 函数原型

```cpp
bool GetRmsNormMaxMinTmpSize(
    const ge::Shape& srcShape, 
    const uint32_t typeSize, 
    uint32_t& maxValue, 
    uint32_t& minValue, 
    const bool isBasicBlock = false
)

bool GetRmsNormTilingInfo(
    const ge::Shape& srcShape, 
    const ge::Shape& originSrcShape, 
    const uint32_t stackBufferByteSize, 
    const uint32_t typeSize, 
    optiling::RmsNormTiling& tiling, 
    const bool isBasicBlock = false
)
```

## 参数说明

### GetRmsNormMaxMinTmpSize 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| srcShape | 输入 | 输入的 shape 信息 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为 half，此处应传入 2 |
| maxValue | 输出 | RmsNorm 接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel 侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为 0 表示计算不需要临时空间。<br>**说明**：maxValue 仅作为参考值，有可能大于 Unified Buffer 剩余空间的大小，该场景下，开发者需要根据 Unified Buffer 剩余空间的大小来选取合适的临时空间大小 |
| minValue | 输出 | RmsNorm 接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为 0 表示计算不需要临时空间 |
| isBasicBlock | 输入 | 是否要使能基本块计算，与 kernel 侧接口一致，默认 false |

### GetRmsNormTilingInfo 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| srcShape | 输入 | 输入的 tensor 的 shape 信息，这里是 H 轴向上 32B 对齐后的 shape。需要保证 srcShape 的 B/S 和 originSrcShape 的 B/S 一致 |
| originSrcShape | 输入 | 输入的原始 shape 信息 |
| stackBufferByteSize | 输入 | 剩余的可供 RmsNorm 接口计算的空间大小，单位为 Byte。通过 `GetRmsNormMaxMinTmpSize` 获取最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为 stackBufferByteSize 传入 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为 half，此处应传入 2 |
| tiling | 输出 | RmsNorm 计算所需 Tiling 信息 |
| isBasicBlock | 输入 | 是否要使能基本块计算，与 kernel 侧接口一致，默认 false。若使能基本块，则需要保证 originSrcShape 的 H 也是 32B 对齐 |

## 返回值说明

- **GetRmsNormMaxMinTmpSize** 返回值为 `true/false`，`true` 表示成功拿到 RmsNorm 接口内部计算需要的最大和最小临时空间大小；`false` 表示获取失败，获取失败情况下，需要检查输入的 shape 是否符合要求

- **GetRmsNormTilingInfo** 返回类型为 `true/false`，`true` 表示成功拿到 RmsNorm 的 Tiling 各项参数值；`false` 表示获取失败，获取失败情况下需要检查输入的 `stackBufferByteSize` 是否满足最小临时空间要求，若开启 `isBasicBlock` 开关，则需要检查输入 shape 是否满足基本块的要求

## 约束说明
无

## 调用示例

### 步骤 1
将 RmsNorm Tiling 结构体参数增加至 TilingData 结构体，作为 TilingData 结构体的一个字段

```cpp
BEGIN_TILING_DATA_DEF(RmsnormCustomTilingData) // 注册一个 tiling 的类，以 tiling 的名字作为入参
TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 添加 tiling 字段，总计算数据量
TILING_DATA_FIELD_DEF(uint32_t, tileNum); // 添加 tiling 字段，每个核上总计算数据分块个数
TILING_DATA_FIELD_DEF(uint32_t, tmpBufSize); // 添加 tiling 字段，临时空间大小
... // 添加其他 tiling 字段
TILING_DATA_FIELD_DEF_STRUCT(RmsNormTiling, rmsnormTilingData); // 将 RmsNormTiling 结构体参数增加至 TilingData 结构体
END_TILING_DATA_DEF;
```

### 步骤 2
Tiling 实现函数中，首先调用 `GetRmsNormMaxMinTmpSize` 接口获取 RmsNorm 接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入 shape、剩余的可供计算的空间大小等信息获取 RmsNorm kernel 侧接口所需 tiling 参数

```cpp
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    RmsNormCustomTilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    // 设置其他 Tiling 参数
    ...
    
    std::vector<int64_t> shapeVec = {2, 16, 64};
    ge::Shape srcShape(shapeVec);
    std::vector<int64_t> oriShapeVec = {2, 16, 64};
    ge::Shape oriSrcShape(oriShapeVec);
    
    // 本样例中仅作为样例说明，通过 GetRmsNormMaxMinTmpSize 获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
    uint32_t minValue = 0;
    uint32_t maxValue = 0;
    AscendC::GetRmsNormMaxMinTmpSize(srcShape, sizeof(half), maxValue, minValue, isBasicBlock);
    tiling.set_tmpBufSize(minValue);
    
    // 获取 RmsNorm Tiling 参数
    AscendC::GetRmsNormTilingInfo(srcShape, oriSrcShape, minValue, sizeof(half), tiling.rmsnormTilingData, false);

    ... // 其他逻辑
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

### 步骤 3
对应的 kernel 侧通过在核函数中调用 `GET_TILING_DATA` 获取 TilingData，继而将 TilingData 中的 RmsNorm Tiling 信息传入 RmsNorm 接口参与计算。完整的 kernel 侧样例请参考 15.1.5.4.7 RmsNorm

```cpp
extern "C" __global__ __aicore__ void rmsnorm_custom(GM_ADDR inputGm, GM_ADDR gammaGm, GM_ADDR outputGm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelRmsNorm op;
    op.Init(inputGm, gammaGm, outputGm, tilingData.totalLength, tilingData.tileNum, tilingData.rmsnormTilingData);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
```
