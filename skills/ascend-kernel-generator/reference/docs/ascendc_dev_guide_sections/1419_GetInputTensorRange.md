##### GetInputTensorRange

## 函数功能

根据算子输入索引获取对应的输入 tensor range 指针。这里的输入索引是指算子实例化后实际的索引，不是原型定义中的索引。

## 函数原型

```cpp
const TensorRange *GetInputTensorRange(const size_t index) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| index | 输入 | 算子输入索引，从0开始计数 |

## 返回值说明

TensorRange 类型指针，定义如下：

```cpp
using TensorRange = Range<Tensor>;
```

index 非法时，返回空指针。

## 约束说明

如果输入没有被设置为数据依赖，调用此接口获取 tensor range 时，只能在 tensor 中获取到正确的 shape、format、datatype 信息，无法获取到真实的 tensor 数据地址（获取到的地址为 nullptr）。

## 调用示例

```cpp
const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
    auto input_tensor_range = context->GetInputTensorRange(0U);
    auto output_shape_range = context->GetOutputShapeRange(0U);
    auto input_tensor = input_tensor_range->GetMax();
    auto shape_data = input_tensor->GetData<int64_t>();
    
    auto shape_size = input_tensor->GetShapeSize();
    // ... output_shape_range推导逻辑
    return GRAPH_SUCCESS;
};
```
