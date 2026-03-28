##### GetRequiredInputTensor

## 函数功能

根据算子原型定义中的输入索引获取对应的必选输入 Tensor 指针。

## 函数原型

```cpp
const Tensor *GetRequiredInputTensor(const size_t ir_index) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| `ir_index` | 输入 | 算子 IR 原型定义中的输入索引，从 0 开始计数。 |

## 返回值说明

输入 Tensor 指针。如果 `ir_index` 非法，则返回空指针。

> 关于 Tensor 类型的定义，请参见 15.2.2.34 Tensor。

## 调用示例

```cpp
ge::graphStatus InferFormatForXXX(InferFormatContext *context) {
  const auto data = context->GetRequiredInputTensor(1U)->GetData<uint8_t>();
  EXPECT_EQ(data[0], 85);
  // ...
}
```
