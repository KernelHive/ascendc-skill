##### GetDropOutMaxMinTmpSize

## 功能说明

用于获取 DropOut Tiling 参数。

## 函数原型

```cpp
uint32_t GetDropOutMaxTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource)
uint32_t GetDropOutMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource)
void GetDropOutMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|------------|------|
| srcShape | 输入 | 输入的 shape 信息。 |
| typeSize | 输入 | 计算的数据类型大小，half=2，float=4。 |
| isReuseSource | 输入 | 预留参数，暂未启用，保持默认值 false 即可。 |
| maxValue | 输出 | 输出 DropOut 接口所需的 tiling 信息（最大临时空间大小）。<br>说明：maxValue 仅作为参考值，有可能大于 Unified Buffer 剩余空间的大小，该场景下，开发者需要根据 Unified Buffer 剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | 输出 DropOut 接口所需的 tiling 信息（最小临时空间大小）。 |

## 返回值说明

- `GetDropOutMaxTmpSize`：返回 DropOut 接口能完成计算所需最大临时空间大小。
- `GetDropOutMinTmpSize`：返回 DropOut 接口能完成计算所需最小临时空间大小。
- `GetDropOutMaxMinTmpSize`：无返回值。

## 约束说明

无

## 调用示例

下文呈现了一个 host 侧调用 `GetDropOutMaxMinTmpSize` 接口的使用示例，通过该接口获取 DropOut 计算所需的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为 Tiling 参数传递到 kernel 侧使用。配套的 kernel 侧使用样例请参考调用示例。

```cpp
#include <vector>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(DropoutCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, firstAxis);
TILING_DATA_FIELD_DEF(uint32_t, srcLastAxis);
TILING_DATA_FIELD_DEF(uint32_t, maskLastAxis);
TILING_DATA_FIELD_DEF(uint32_t, tmpBufferSize);
END_TILING_DATA_DEF;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // Input source shapes.
    int64_t firstAxis = 16;
    int64_t srcLastAxis = 64;
    int64_t maskLastAxis = 64;

    std::vector<int64_t> srcDims = {firstAxis, srcLastAxis, maskLastAxis};

    uint32_t typeSize = 2;
    ge::Shape shape(srcDims);
    uint32_t minValue = 0;
    uint32_t maxValue = 0;
    AscendC::GetDropOutMaxMinTmpSize(shape, typeSize, false, maxValue, minValue);

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t tailSize = 0; // ub剩余空间大小
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, tailSize); // 本样例中使用完整的ub空间，实际情况下tailSize需要减掉用户已使用的ub空间
    auto tmpSize = tailSize >= maxValue ? maxValue : tailSize;

    DropoutCustomTilingData tiling;
    tiling.set_firstAxis(firstAxis);
    tiling.set_srcLastAxis(srcLastAxis);
    tiling.set_maskLastAxis(maskLastAxis);
    tiling.set_tmpBufferSize(tmpSize);
    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```
