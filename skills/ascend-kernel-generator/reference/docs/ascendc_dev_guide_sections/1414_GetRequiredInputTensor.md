##### GetRequiredInputTensor

## 函数功能

根据算子原型定义中的输入索引获取对应的必选输入 tensor 指针。

## 函数原型

```cpp
const Tensor *GetRequiredInputTensor(const size_t ir_index) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| ir_index | 输入 | 必选输入在算子 IR 原型定义中的索引，从 0 开始计数 |

## 返回值说明

返回指定输入的 tensor 指针，若输入的 ir_index 非法，返回空指针。

关于 Tensor 类型的定义，请参见 15.2.2.34 Tensor。

## 约束说明

仅在设置数据依赖时可以获取 tensor 的数据地址。如果输入没有被设置为数据依赖，调用此接口获取 tensor 时，只能在 tensor 中获取到正确的 shape、format、datatype 信息，无法获取到真实的 tensor 数据地址（获取到的地址为 nullptr）。

## 调用示例

```cpp
ge::graphStatus InferShapeForXXX(InferShapeContext *context) {
  auto in_shape = context->GetRequiredInputTensor(2);
  // ...
}
```
