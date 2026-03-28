##### BatchNorm Tiling

## 功能说明

BatchNorm Tiling API 用于获取 BatchNorm kernel 计算时所需的 Tiling 参数。获取 Tiling 参数主要分为如下两步：

### 1. 获取临时空间大小

通过 `GetBatchNormMaxMinTmpSize` 获取 BatchNorm 接口计算所需最大和最小临时空间大小。

kernel 侧 BatchNorm 接口的计算需要开发者预留/申请临时空间，`GetBatchNormMaxMinTmpSize` 用于在 host 侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为 Tiling 参数传递到 kernel 侧使用。

- 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小
- 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel 侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请

### 2. 获取 Tiling 参数

通过 `GetBatchNormNDTilingInfo` 获取 BatchNorm kernel 侧接口所需 tiling 参数。

BatchNorm Tiling 结构体的定义如下，开发者无需关注该 tiling 结构的具体信息，只需要传递到 kernel 侧，传入 BatchNorm 高阶 API 接口，直接进行使用即可。

```cpp
struct BatchNormTiling {
    uint32_t originalBLength = 0;
    uint32_t meanVarSize = 0;
    uint32_t meanTmpTensorPos = 0;
    uint32_t varianceTmpTensorPos = 0;
    uint32_t tmpBufSize = 0;
    uint32_t oneTmpSize = 0;
    uint32_t firstTmpStartPos = 0;
    uint32_t secondTmpStartPos = 0;
    uint32_t thirdTmpStartPos = 0;
    uint32_t loopRound = 0;
    uint32_t inputTailSize = 0;
    uint32_t inputTailPos = 0;
    uint32_t meanVarTailSize = 0;
    uint32_t meanVarTailPos = 0;
    uint32_t bshCurLength = 0;
    uint32_t shCurLength = 0;
    float firstDimValueBack = 0;
    uint32_t castHalfRepStride = 0;
    uint32_t shCurLengthBlockNum = 0;
    uint32_t castHalfOutRepStride = 0;
};
```

## 函数原型

```cpp
bool GetBatchNormMaxMinTmpSize(const ge::Shape& srcShape, 
                              const ge::Shape& originSrcShape, 
                              const uint32_t typeSize, 
                              const bool isReuseSource, 
                              uint32_t& maxValue,
                              uint32_t& minValue, 
                              const bool isBasicBlock = false)
```

```cpp
bool GetBatchNormNDTilingInfo(const ge::Shape& srcShape, 
                             const ge::Shape& originSrcShape, 
                             const uint32_t stackBufferByteSize, 
                             const uint32_t typeSize, 
                             const bool isReuseSource, 
                             optiling::BatchNormTiling& tilling, 
                             const bool isBasicBlock = false)
```

## 参数说明

### GetBatchNormMaxMinTmpSize 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| srcShape | 输入 | 输入数据 inputX 的 shape 信息 [B, S, H]，S*H 需要 32B 对齐 |
| originSrcShape | 输入 | 输入数据 inputX 的 origin shape 信息 [originB, originS, originH] |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为 half，此处应传入 2 |
| isReuseSource | 输入 | 中间变量是否能够复用输入内存。该参数预留，传入默认值 false 即可 |
| maxValue | 输出 | BatchNorm 接口能完成计算所需的最大临时空间大小，超出 max 的空间不会被该接口使用。在 min-max 范围内，预留/申请空间越大，接口计算性能越好。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。maxValue 为 0 表示计算不需要临时空间 |
| minValue | 输出 | BatchNorm 接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于 min 的数值。最小空间为 0 表示计算不需要临时空间 |
| isBasicBlock | 输入 | 是否使能基本块，与 BatchNorm 接口一致 |

**说明**：maxValue 仅作为参考值，有可能大于 Unified Buffer 剩余空间的大小，该场景下，开发者需要根据 Unified Buffer 剩余空间的大小来选取合适的临时空间大小。

### GetBatchNormNDTilingInfo 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| srcShape | 输入 | 输入数据 inputX 的 shape 信息 [B, S, H]，S*H 需要 32B 对齐 |
| originSrcShape | 输入 | 输入数据 inputX 的 origin shape 信息 [originB, originS, originH] |
| stackBufferByteSize | 输入 | 可供 BatchNorm 接口使用的空间大小，单位 Byte |
| typeSize | 输入 | 输入数据类型的字节大小 |
| isReuseSource | 输入 | 中间变量是否能够复用输入内存。该参数预留，传入默认值 false 即可 |
| tilling | 输出 | 输入数据的切分信息 |
| isBasicBlock | 输入 | 是否使能基本块，与 BatchNorm 接口一致 |

## 返回值说明

- `GetBatchNormMaxMinTmpSize` 返回值为 true/false，true 表示成功拿到 BatchNorm 接口内部计算需要的最大和最小临时空间大小；false 表示获取失败
- `GetBatchNormNDTilingInfo` 返回类型为 true/false，true 表示成功拿到 BatchNorm 的 Tiling 各项参数值；false 表示获取失败

## 约束说明

无

## 调用示例

如下样例介绍了 host 侧获取 Tiling 参数的流程以及该参数如何在 kernel 侧使用。样例中输入 Tensor 的 shape 大小为 [16, 16, 16]，输入的数据类型为 half。

### 步骤 1：注册 Tiling 结构体

将 BatchNormTiling 结构体参数增加至 TilingData 结构体，作为 TilingData 结构体的一个字段。

```cpp
BEGIN_TILING_DATA_DEF(TilingData) // 注册一个 tiling 的类，以 tiling 的名字作为入参
    TILING_DATA_FIELD_DEF(uint32_t, tileNum); // 添加 tiling 字段，每个核上总计算数据分块个数
    TILING_DATA_FIELD_DEF(uint32_t, bLength); // 添加 tiling 字段，输入 shape 的 b 维度长度
    TILING_DATA_FIELD_DEF(uint32_t, sLength); // 添加 tiling 字段，输入 shape 的 s 维度长度
    TILING_DATA_FIELD_DEF(uint32_t, hLength); // 添加 tiling 字段，输入 shape 的 h 维度长度
    TILING_DATA_FIELD_DEF(uint32_t, originalBLength); // 添加 tiling 字段，输入 shape 原始 b 维度长度
    ... // 添加其他 tiling 字段
    TILING_DATA_FIELD_DEF_STRUCT(BatchNormTiling, batchNormTilingData); // 将 BatchNormTiling 结构体参数增加至 TilingData 结构体
END_TILING_DATA_DEF;
```

### 步骤 2：Tiling 实现函数

在 Tiling 实现函数中，首先调用 `GetBatchNormMaxMinTmpSize` 接口获取 BatchNorm 接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入 shape、剩余的可供计算的空间大小等信息获取 BatchNorm kernel 侧接口所需 tiling 参数。

```cpp
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_tileNum(TILE_NUM);
    // 设置其他 Tiling 参数
    ...
    
    std::vector<int64_t> shapeVec = {16, 16, 16}; // {b,s,h}
    std::vector<int64_t> originShapeVec = {15, 16, 16}; // {originB,originS,originH}
    ge::Shape srcShape(shapeVec);
    ge::Shape originSrcShape(originShapeVec);
    uint32_t minSize = 0;
    uint32_t maxSize = 0;
    
    // 本样例中仅作为样例说明，通过 GetBatchNormMaxMinTmpSize 获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
    AscendC::GetBatchNormMaxMinTmpSize(srcShape, originSrcShape, sizeof(half), false, maxSize, minSize, false);
    
    // 获取 BatchNorm Tiling 参数
    AscendC::GetBatchNormNDTilingInfo(srcShape, originSrcShape, minSize, sizeof(half), false, tiling.batchNormTilingData, false);
    
    ... // 其他逻辑
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

### 步骤 3：Kernel 侧调用

对应的 kernel 侧通过在核函数中调用 `GET_TILING_DATA` 获取 TilingData，继而将 TilingData 中的 BatchNormTiling 信息传入 BatchNorm 接口参与计算。

```cpp
extern "C" __global__ __aicore__ void func_custom(GM_ADDR inputX_gm, 
                                                  GM_ADDR gamm_gm, 
                                                  GM_ADDR beta_gm, 
                                                  GM_ADDR output_gm, 
                                                  GM_ADDR outputMean_gm, 
                                                  GM_ADDR outputVariance_gm, 
                                                  GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelBatchnorm<half, false, false> op;
    op.Init(inputX_gm, gamm_gm, beta_gm, output_gm, outputMean_gm, outputVariance_gm, tilingData.batchNormTilingData);

    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
```
