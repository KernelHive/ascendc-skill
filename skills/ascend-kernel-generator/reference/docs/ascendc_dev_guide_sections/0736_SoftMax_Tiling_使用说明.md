###### SoftMax Tiling 使用说明

Ascend C 提供一组 SoftMax Tiling API，方便用户获取 SoftMax kernel 计算时所需的 Tiling 参数。阅读本节之前，请先参考 Tiling 实现了解 Tiling 实现基本流程。

获取 Tiling 参数主要分为如下两步：

1. 获取 SoftMax 接口计算所需最小和最大临时空间大小。注意该步骤不是必须的，只是作为一个参考，供合理分配计算空间。
2. 获取输入 SoftMax kernel 侧接口所需 tiling 参数，需要传入输入 shape、剩余的可供 softmax 接口计算的空间大小和计算的数据类型大小。

SoftMax Tiling 结构体的定义如下，开发者无需关注该 tiling 结构的具体信息，只需要传递到 kernel 侧，传入 SoftMax 高阶 API 接口，直接进行使用即可。

```cpp
struct SoftMaxTiling {
    uint32_t srcM = 0;
    uint32_t srcK = 0;
    uint32_t srcSize = 0;
    uint32_t outMaxM = 0;
    uint32_t outMaxK = 0;
    uint32_t outMaxSize = 0;
    uint32_t splitM = 0;
    uint32_t splitK = 0;
    uint32_t splitSize = 0;
    uint32_t reduceM = 0;
    uint32_t reduceK = 0;
    uint32_t reduceSize = 0;
    uint32_t rangeM = 0;
    uint32_t tailM = 0;
    uint32_t tailSplitSize = 0;
    uint32_t tailReduceSize = 0;
};
```

- 对于 SoftMax/SimpleSoftMax 请参考 SoftMax/SimpleSoftMax Tiling；
- 对于 SoftmaxFlash 请参考 SoftmaxFlash Tiling 接口；
- 对于 SoftmaxGrad 请参考 SoftmaxGrad Tiling 接口；
- 对于 SoftmaxFlashV2 请参考 SoftmaxFlashV2 Tiling 接口；
- 判断 SoftMaxTiling 是否为基本块 Tiling 请参考 IsBasicBlockInSoftMax。

## 调用示例

如下样例介绍了使用 SoftMax 高阶 API 时 host 侧获取 Tiling 参数的流程以及该参数如何在 kernel 侧使用。样例中输入 Tensor 的 shape 大小为 `[320,64]`，输入的数据类型为 `half`。

### 步骤 1

将 SoftMaxTiling 结构体参数增加至 TilingData 结构体，作为 TilingData 结构体的一个字段。

```cpp
BEGIN_TILING_DATA_DEF(TilingData) // 注册一个 tiling 的类，以 tiling 的名字作为入参
TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 添加 tiling 字段，总计算数据量
TILING_DATA_FIELD_DEF(uint32_t, tileNum); // 添加 tiling 字段，每个核上总计算数据分块个数
... // 添加其他 tiling 字段
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData); // 将 SoftMaxTiling 结构体参数增加至 TilingData 结构体
END_TILING_DATA_DEF;
```

### 步骤 2

Tiling 实现函数中，首先调用 `GetSoftMaxMaxTmpSize`/`GetSoftMaxMinTmpSize` 接口获取 SoftMax 接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小；然后根据输入 shape、剩余的可供计算的空间大小等信息获取 SoftMax kernel 侧接口所需 tiling 参数。

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
    // 设置其他 Tiling 参数
    ...
    std::vector<int64_t> shapeVec = {320,64};
    ge::Shape srcShape(shapeVec);
    // 本样例中仅作为样例说明，通过 GetSoftMaxMinTmpSize 获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
    const uint32_t localWorkSpaceSize = AscendC::GetSoftMaxMinTmpSize(srcShape, sizeof(half), false);
    // 获取 SoftMax Tiling 参数
    AscendC::SoftMaxTilingFunc(srcShape, sizeof(half), localWorkSpaceSize, tiling.softmaxTilingData);
    ... // 其他逻辑
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

### 步骤 3

对应的 kernel 侧通过在核函数中调用 `GET_TILING_DATA` 获取 TilingData，继而将 TilingData 中的 SoftMax Tiling 信息传入 SoftMax 接口参与计算。完整的 kernel 侧样例请参考调用示例。

```cpp
extern "C" __global__ __aicore__ void func_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelFunc op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum, tilingData.softmaxTiling);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
```
