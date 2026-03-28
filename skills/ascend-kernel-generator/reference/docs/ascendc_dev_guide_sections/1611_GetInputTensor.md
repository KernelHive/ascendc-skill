##### GetInputTensor

## 函数功能

根据算子输入索引获取对应的输入 tensor 指针。这里的输入索引是指算子实例化后实际的索引，不是原型定义中的索引。

## 函数原型

```cpp
const Tensor *GetInputTensor(const size_t index) const
```

## 参数说明

| 参数   | 输入/输出 | 说明                         |
|--------|-----------|------------------------------|
| index  | 输入      | 算子输入索引，从 0 开始计数。 |

## 返回值说明

指定的输入 tensor 指针，当输入 index 非法时返回空指针。

关于 Tensor 类型的定义，请参见 15.2.2.34 Tensor。

## 约束说明

仅在设置数据依赖时可以获取 tensor 的数据地址。如果输入没有被设置为数据依赖，调用此接口获取 tensor 时，只能在 tensor 中获取到正确的 shape、format、datatype 信息，无法获取到真实的 tensor 数据地址（获取到的地址为 nullptr）。

## 调用示例

```cpp
ge::graphStatus Tiling4ReduceCommon(TilingContext* context) {
    auto in_shape = context->GetInputShape(0);
    GE_ASSERT_NOTNULL(in_shape);
    auto axes_tensor = context->GetInputTensor(1);
    // ...
}
```
