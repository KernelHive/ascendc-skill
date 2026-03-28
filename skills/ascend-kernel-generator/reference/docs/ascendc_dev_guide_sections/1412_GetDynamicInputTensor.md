##### GetDynamicInputTensor

## 函数功能

根据算子原型定义中的输入索引获取对应的动态输入 tensor 指针。

## 函数原型

```cpp
const Tensor *GetDynamicInputTensor(const size_t ir_index, const size_t relative_index) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| ir_index | 输入 | 算子 IR 原型定义中的输入索引，从 0 开始计数 |
| relative_index | 输入 | 该输入实例化后的相对 index，例如某个 DYNAMIC_INPUT 实例化了 3 个输入，那么 relative_index 的有效范围是 [0,2] |

## 返回值说明

指定的输入 tensor 指针，ir_index 或 relative_index 非法时，返回空指针。

关于 Tensor 类型的定义，请参见 15.2.2.34 Tensor。

## 约束说明

仅在设置数据依赖时可以获取 tensor 的数据地址。如果输入没有被设置为数据依赖，调用此接口获取 tensor 时，只能在 tensor 中获取到正确的 shape、format、datatype 信息，无法获取到真实的 tensor 数据地址（获取到的地址为 nullptr）。

## 调用示例

```cpp
ge::graphStatus InferShapeForXXX(InferShapeContext *context) {
    auto in_shape = context->GetInputShape(0);
    GE_ASSERT_NOTNULL(in_shape);
    auto axes_tensor_10 = context->GetDynamicInputTensor(1, 0);
    auto axes_tensor_11 = context->GetDynamicInputTensor(1, 1);
    // ...
}
```
