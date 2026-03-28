#### 通过 TilingData 传递属性信息

如果算子包含属性信息，该属性信息可以通过 TilingData 传递到 kernel 侧，参与 kernel 侧算子核函数的计算。

以 ReduceMaxCustom 算子为例，该算子用于对输入数据按维度 `dim` 返回最大值，并且返回索引。ReduceMaxCustom 算子有两个属性：

- `reduceDim`：表示按照哪一个维度进行 reduce 操作
- `isKeepDim`：表示是否需要保持输出的维度与输入一样

本样例仅支持对最后一维做 reduce 操作，输入数据类型为 half。

## 1. ReduceMaxCustom 算子 TilingData 的定义

这里我们重点关注 `reduceAxisLen`。参数 `reduceAxisLen` 表示获取 `reduceDim` 轴的长度，这里也就是最后一维的长度。该参数后续会通过 TilingData 传递到 kernel 侧参与计算。

```cpp
#ifndef REDUCE_MAX_CUSTOM_TILING_H
#define REDUCE_MAX_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReduceMaxTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, reduceAxisLen); // 添加tiling字段，reduceDim轴的长度
  // 其他TilingData参数的定义
  ...
END_TILING_DATA_DEF;

// 注册算子tilingdata类到对应的ReduceMaxCustom算子
REGISTER_TILING_DATA_CLASS(ReduceMaxCustom, ReduceMaxTilingData)
}
#endif // REDUCE_MAX_CUSTOM_TILING_H
```

## 2. ReduceMaxCustom 算子的 Tiling 实现

这里我们重点关注属性信息通过 TilingData 传递的过程：

1. 通过 TilingContext 上下文从 attr 获取 `reduceDim` 属性值
2. 根据 `reduceDim` 属性值获取 `reduceDim` 轴的长度并设置到 TilingData 中

```cpp
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  ReduceMaxTilingData tiling;

  // 从attr获取reduceDim属性值，因为reduceDim是第一个属性，所以GetAttrPointer传入的索引值为0
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  const uint32_t* reduceDim = attrs->GetAttrPointer<uint32_t>(0);

  // 获取reduceDim轴的长度
  const gert::StorageShape* xShapePtr = context->GetInputShape(0);
  const gert::Shape& xShape = xShapePtr->GetStorageShape();
  const uint32_t reduceAxisLen = xShape.GetDim(*reduceDim);

  // 计算TilingData中除了reduceAxisLen之外其他成员变量的值
  ...

  // 将reduceAxisLen设置到tiling结构体中，传递到kernel函数使用
  tiling.set_reduceAxisLen(reduceAxisLen);

  // 设置TilingData中除了reduceAxisLen之外其他成员变量的值
  ...

  // TilingData序列化保存
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  ...

  return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```
