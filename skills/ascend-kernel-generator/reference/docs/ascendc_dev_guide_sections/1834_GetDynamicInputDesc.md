##### GetDynamicInputDesc

## 函数功能

根据 `name` 和 `index` 的组合获取算子动态 Input 的 TensorDesc。

## 函数原型

```cpp
TensorDesc GetDynamicInputDesc(const std::string &name, uint32_t index) const
TensorDesc GetDynamicInputDesc(const char_t *name, uint32_t index) const
```

> **须知**
>
> 数据类型为 `string` 的接口后续版本会废弃，建议使用数据类型为非 `string` 的接口。

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| `name` | 输入 | 算子动态 Input 的名称。 |
| `index` | 输入 | 算子动态 Input 编号，编号从 0 开始。 |

## 返回值

获取 TensorDesc 成功，则返回算子动态 Input 的 TensorDesc；获取失败，则返回 TensorDesc 默认构造的对象，其中主要设置：

- `DataType` 为 `DT_FLOAT`（表示 float 类型）
- `Format` 为 `FORMAT_NCHW`（表示 NCHW）

## 异常处理

无。

## 约束说明

无。
