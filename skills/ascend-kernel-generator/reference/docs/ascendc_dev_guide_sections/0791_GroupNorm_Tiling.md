##### GroupNorm Tiling

## 功能说明

GroupNorm Tiling API 用于获取 GroupNorm kernel 计算时所需的 Tiling 参数。获取 Tiling 参数主要分为如下两步：

1. **通过 `GetGroupNormMaxMinTmpSize` 获取 GroupNorm 接口计算所需最大和最小临时空间大小**

   kernel 侧 GroupNorm 接口的计算需要开发者预留/申请临时空间，`GetGroupNormMaxMinTmpSize` 用于在 host 侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为 Tiling 参数传递到 kernel 侧使用。

   - 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
   - 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel 侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

2. **通过 `GetGroupNormNDTilingInfo` 获取 GroupNorm kernel 侧接口所需 tiling 参数**

   GroupNorm Tiling 结构体的定义如下，开发者无需关注该 tiling 结构的具体信息，只需要传递到 kernel 侧，传入 GroupNorm 高阶 API 接口，直接进行使用即可。

```cpp
struct GroupNormTiling {
    uint32_t n = 0;
    uint32_t c = 0;
    uint32_t hw = 0;
    uint32_t g = 0;
    uint32_t d = 0;
    uint32_t hwAlignSize = 0;
    uint32_t dhwAlignSize = 0;
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
    float factor = 0;
    bool smallShape = 0;
};
```

## 函数原型

```cpp
void GetGroupNormMaxMinTmpSize(
    const ge::Shape& srcShape,
    const uint32_t typeSize,
    const bool isReuseSource,
    const uint32_t groupNum,
    uint32_t& maxValue,
    uint32_t& minValue
);

void GetGroupNormNDTilingInfo(
    const ge::Shape& srcShape,
    const uint32_t stackBufferSize,
    const uint32_t typeSize,
    const bool isReuseSource,
    const uint32_t groupNum,
    optiling::GroupNormTiling& tiling
);
```

## 参数说明

### GetGroupNormMaxMinTmpSize 接口参数列表

| 参数名称 | 输入/输出 | 功能说明 |
|----------|-----------|----------|
| srcShape | 输入 | 输入数据 inputX 的 shape 信息 [N, C, H, W] |
| typeSize | 输入 | 输入数据 inputX 的数据类型大小，单位为字节。比如输入的数据类型为 half，此处应传入 2 |
| isReuseSource | 输入 | 中间变量是否能够复用输入内存。该参数预留，传入默认值 false 即可 |
| groupNum | 输入 | 在 C 维度上的分组数 |
| maxValue | 输出 | 输出 GroupNorm 接口所需的 tiling 信息（最大临时空间大小）。GroupNorm 接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel 侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。<br>**说明：** maxValue 仅作为参考值，有可能大于 Unified Buffer 剩余空间的大小，该场景下，开发者需要根据 Unified Buffer 剩余空间的大小来选取合适的临时空间大小 |
| minValue | 输出 | 输出 GroupNorm 接口所需的 tiling 信息（最小临时空间大小）。GroupNorm 接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值 |

### GetGroupNormNDTilingInfo 接口参数列表

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| srcShape | 输入 | 输入数据 inputX 的 shape 信息 [N, C, H, W] |
| stackBufferSize | 输入 | 可供 GroupNorm 接口使用的空间大小，单位 Byte |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为 half，此处应传入 2 |
| isReuseSource | 输入 | 是否可以复用 inputX 的内存空间 |
| groupNum | 输入 | 在 C 维度上的分组数 |
| tiling | 输出 | 输入数据的切分信息 |

## 返回值说明

无

## 约束说明

无

## 调用示例

如下样例介绍了 host 侧获取 Tiling 参数的流程以及该参数如何在 kernel 侧使用。样例中输入 Tensor 的 shape 大小为 [2, 16, 8, 8]，输入的数据类型为 half。

### 步骤 1：将 GroupNormTiling 结构体参数增加至 TilingData 结构体

将 GroupNormTiling 结构体参数作为 TilingData 结构体的一个字段。

```cpp
BEGIN_TILING_DATA_DEF(TilingData) // 注册一个 tiling 的类，以 tiling 的名字作为入参
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, c);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, w);
    TILING_DATA_FIELD_DEF(uint32_t, group);
    // 添加其他 tiling 字段
    ...
    TILING_DATA_FIELD_DEF_STRUCT(GroupNormTiling, GroupNormTilingData); // 将 GroupNormTiling 结构体参数增加至 TilingData 结构体
END_TILING_DATA_DEF;
```

### 步骤 2：Tiling 实现函数

在 Tiling 实现函数中，首先调用 `GetGroupNormMaxMinTmpSize` 接口获取 GroupNorm 接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入 shape、剩余的可供计算的空间大小等信息获取 GroupNorm kernel 侧接口所需 tiling 参数。

```cpp
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    TilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_tileNum(TILE_NUM);
    // 设置其他 Tiling 参数
    ...
    
    std::vector<int64_t> shapeVec = {2, 16, 8, 8}; // {n, c, h, w}
    ge::Shape srcShape(shapeVec);
    uint32_t groupNum = 4;
    uint32_t minSize = 0;
    uint32_t maxSize = 0;
    
    // 本样例中仅作为样例说明，通过 GetGroupNormMaxMinTmpSize 接口获取 GroupNorm 接口能完成计算所需最大/最小临时空间大小，开发者可以根据该范围结合实际的内存使用情况设置合适的空间大小
    AscendC::GetGroupNormMaxMinTmpSize(srcShape, sizeof(half), false, groupNum, maxSize, minSize);
    
    // 获取 GroupNorm Tiling 参数
    AscendC::GetGroupNormNDTilingInfo(srcShape, maxSize, sizeof(half), false, groupNum, tiling.groupNormTilingData);
    
    ... // 其他逻辑
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

### 步骤 3：Kernel 侧调用

对应的 kernel 侧通过在核函数中调用 `GET_TILING_DATA` 获取 TilingData，继而将 TilingData 中的 GroupNorm Tiling 信息传入 GroupNorm 接口参与计算。

```cpp
extern "C" __global__ __aicore__ void groupnorm_custom(
    GM_ADDR inputX_gm, 
    GM_ADDR gamm_gm, 
    GM_ADDR beta_gm, 
    GM_ADDR output_gm, 
    GM_ADDR outputMean_gm, 
    GM_ADDR outputVariance_gm, 
    GM_ADDR tiling
) {
    GET_TILING_DATA(tilingData, tiling);
    KernelGroupNorm<half, false> op;
    op.Init(inputX_gm, gamm_gm, beta_gm, output_gm, outputMean_gm, outputVariance_gm, tilingData.groupNormTilingData);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
```
