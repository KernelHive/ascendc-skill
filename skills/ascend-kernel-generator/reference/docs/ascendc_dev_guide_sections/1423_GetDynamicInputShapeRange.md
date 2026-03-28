##### GetDynamicInputShapeRange

## 函数功能

根据算子原型定义中的输入索引获取对应的动态输入 shape range 指针。

## 函数原型

```cpp
const Range<Shape> *GetDynamicInputShapeRange(const size_t ir_index, const size_t relative_index) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| ir_index | 输入 | 动态输入在算子 IR 原型定义中的索引，从 0 开始计数 |
| relative_index | 输入 | 该输入实例化后的相对 index，例如某个 DYNAMIC_INPUT 实例化了 3 个输入，那么 relative_index 的取值范围是 [0,2] |

## 返回值说明

shape range 指针，ir_index 或 relative_index 非法时，返回空指针。

## 约束说明

无。

## 调用示例

```cpp
const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_shape_range = context->GetDynamicInputShapeRange(0U, 0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
    output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
    return GRAPH_SUCCESS;
};
```
