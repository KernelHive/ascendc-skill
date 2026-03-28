##### OutputTensorDesc

## 函数功能

设置 Tensor 描述信息，用于构造 KernelContext 的基类 ExtendedKernelContext 中的 ComputeNodeInfo 信息。

## 函数原型

```cpp
OpKernelContextBuilder &OutputTensorDesc(
    size_t index,
    ge::DataType dtype,
    ge::Format origin_format,
    ge::Format storage_format,
    const gert::ExpandDimsType &expand_dims_type = {}
)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| `index` | 输入 | 算子输入实例索引 |
| `dtype` | 输入 | 输出 Tensor 的数据类型 |
| `origin_format` | 输入 | 输出 Tensor 的原始格式 |
| `storage_format` | 输入 | 输出 Tensor 的存储格式 |
| `expand_dims_type` | 输入 | 输出 Tensor 的补维规则 ExpandDimsType，默认值为 `{}` |

## 返回值说明

OpKernelContextBuilder 对象引用，用于链式调用。

## 约束说明

在调用 Build 方法之前，必须调用该接口，否则构造出的 KernelContext 将包含未定义数据。
