##### GetInputDesc

## 函数功能

根据算子 Input 名称或 Input 索引获取算子 Input 的 TensorDesc。

## 函数原型

```cpp
TensorDesc GetInputDesc(const std::string &name) const
TensorDesc GetInputDescByName(const char_t *name) const
TensorDesc GetInputDesc(uint32_t index) const
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| name   | 输入      | 算子 Input 名称。<br>当无此算子 Input 名称时，则返回 TensorDesc 默认构造的对象，其中，主要设置 DataType 为 DT_FLOAT（表示 float 类型），Format 为 FORMAT_NCHW（表示 NCHW）。 |
| index  | 输入      | 算子 Input 索引。<br>当无此算子 Input 索引时，则返回 TensorDesc 默认构造的对象，其中，主要设置 DataType 为 DT_FLOAT（表示 float 类型），Format 为 FORMAT_NCHW（表示 NCHW）。 |

## 返回值

算子 Input 的 TensorDesc。

## 异常处理

无。

## 约束说明

无。
