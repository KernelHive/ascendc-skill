##### GetOutputShape

## 函数功能

根据算子输出索引获取对应的输出 Shape 指针。这里的输出索引是指算子实例化后实际的索引，不是原型定义中的索引。

## 函数原型

```cpp
Shape *GetOutputShape(const size_t index)
```

## 参数说明

| 参数   | 输入/输出 | 说明                           |
|--------|-----------|--------------------------------|
| index  | 输入      | 算子输出索引，从 0 开始计数。 |

## 返回值说明

返回指定的输出 Shape 指针，输入 index 非法时，返回空指针。

关于 Shape 类型的定义，请参见 15.2.2.27 Shape。

## 约束说明

无。

## 调用示例

```cpp
ge::graphStatus InferShapeForReshape(InferShapeContext *context) {
  const gert::Shape *x_shape = context->GetInputShape(0); // 获取第 0 个输入的 shape
  const gert::Tensor *shape_tensor = context->GetInputTensor(1); // 获取第 1 个输入的 tensor 数据依赖
  gert::Shape *output_shape = context->GetOutputShape(0);
  if (x_shape == nullptr || shape_tensor == nullptr || output_shape == nullptr) {
    // 防御式编程，不应该出现的场景，打印错误并返回失败
    return ge::GRAPH_FAILED;
  }

  // ...
}
```
