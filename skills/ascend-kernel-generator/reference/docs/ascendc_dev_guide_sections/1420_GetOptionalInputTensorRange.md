##### GetOptionalInputTensorRange

## 功能

根据算子原型定义中的输入索引获取对应的可选输入 tensor range 指针。

## 函数原型

```cpp
const TensorRange *GetOptionalInputTensorRange(const size_t ir_index) const
```

## 参数说明

| 参数     | 输入/输出 | 说明                                       |
|----------|-----------|--------------------------------------------|
| ir_index | 输入      | 算子 IR 原型定义中的输入索引，从 0 开始计数 |

## 返回值

- 返回 `TensorRange` 类型指针，定义如下：

  ```cpp
  using TensorRange = Range<Tensor>;
  ```

- 如果 `ir_index` 非法，或该 INPUT 没有实例化时，返回空指针。

## 约束

如果输入没有被设置为数据依赖，调用此接口获取 tensor range 时，只能在 tensor 中获取到正确的 shape、format、datatype 信息，无法获取到真实的 tensor 数据地址（获取到的地址为 `nullptr`）。

## 调用示例

```cpp
const auto infer_shape_range_func = [](gert::InferShapeRangeContext *context) -> graphStatus {
  auto input_shape_range = context->GetOptionalInputTensorRange(0U);
  auto output_shape_range = context->GetOutputShapeRange(0U);
  *output_shape_range->GetMin() = input_shape_range->GetMin()->GetStorageShape();
  *output_shape_range->GetMax() = input_shape_range->GetMax()->GetStorageShape();
  return GRAPH_SUCCESS;
};
```
