##### GetInputShapeRange

## 函数功能

根据算子输入索引获取对应的输入 shape range 指针。这里的输入索引是指算子实例化后实际的索引，不是原型定义中的索引。

## 函数原型

```cpp
const Range<Shape> *GetInputShapeRange(const size_t index) const
```

## 参数说明

| 参数  | 输入/输出 | 说明 |
|-------|-----------|------|
| index | 输入      | 算子输入索引，从0开始计数 |

## 返回值说明

输入 shape range 指针，index 非法时，返回空指针。

## 约束说明

无。

## 调用示例

```cpp
const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_shape_range = context->GetInputShapeRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
    output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
    return GRAPH_SUCCESS;
};
```
