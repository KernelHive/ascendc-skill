##### OutputTensorDesc

## 函数功能

设置算子输出的 Tensor 描述信息，用于构造 InferShapeContext 的基类 ExtendedKernelContext 中的 ComputeNodeInfo 等信息。

## 函数原型

```cpp
OpInferShapeContextBuilder &OutputTensorDesc(
    size_t index,
    ge::DataType dtype,
    ge::Format origin_format,
    ge::Format storage_format,
    const gert::ExpandDimsType &expand_dims_type = {}
)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| index | 输入 | 输出的索引，对应的是算子输出实例索引 |
| dtype | 输入 | 输出 Tensor 的数据类型 |
| origin_format | 输入 | 输出 Tensor 的原始格式 |
| storage_format | 输入 | 输出 Tensor 的存储格式 |
| expand_dims_type | 输入 | 输出 Tensor 的补维规则 ExpandDimsType |

## 返回值说明

OpInferShapeContextBuilder 对象本身，用于链式调用。

## 约束说明

在调用 Build 方法之前，必须调用该接口，否则构造出的 InferShapeContext 将包含未定义数据。
