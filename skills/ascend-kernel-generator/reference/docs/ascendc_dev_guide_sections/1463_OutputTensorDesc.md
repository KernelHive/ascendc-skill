##### OutputTensorDesc

## 函数功能

设置输出 Tensor 描述信息，用于构造 `InferDataTypeContext` 的基类 `ExtendedKernelContext` 中的 `ComputeNodeInfo` 信息。无需设置输出数据类型信息，输出数据类型由算子根据输入数据类型计算推导得到。

## 函数原型

```cpp
OpInferDataTypeContextBuilder &OutputTensorDesc(
    size_t index,
    ge::Format origin_format,
    ge::Format storage_format,
    const gert::ExpandDimsType &expand_dims_type = {}
)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| `index` | 输入 | 算子输出实例索引。 |
| `origin_format` | 输入 | 输出 Tensor 的原始格式。 |
| `storage_format` | 输入 | 输出 Tensor 的存储格式。 |
| `expand_dims_type` | 输入 | 输出 Tensor 的补维规则 `ExpandDimsType`，默认值为 `{}`。 |

## 返回值说明

`OpInferDataTypeContextBuilder` 对象本身，用于链式调用。

## 约束说明

在调用 `Build` 方法之前，必须调用该接口，否则构造出的 `InferDataTypeContext` 将包含未定义数据。
