##### SetAttrValue

## 函数功能

设置属性的取值。泛型类，承载具体的属性类型。

## 函数原型

```cpp
graphStatus SetAttrValue(const int64_t &attr_value) const
graphStatus SetAttrValue(const float32_t &attr_value) const
graphStatus SetAttrValue(const AscendString &attr_value) const
graphStatus SetAttrValue(const bool &attr_value) const
graphStatus SetAttrValue(const Tensor &attr_value) const
graphStatus SetAttrValue(const std::vector<int64_t> &attr_value) const
graphStatus SetAttrValue(const std::vector<float32_t> &attr_value) const
graphStatus SetAttrValue(const std::vector<AscendString> &attr_values) const
graphStatus SetAttrValue(const std::vector<bool> &attr_value) const
graphStatus SetAttrValue(const std::vector<Tensor> &attr_value) const
graphStatus SetAttrValue(const std::vector<std::vector<int64_t>> &attr_value) const
graphStatus SetAttrValue(const std::vector<ge::DataType> &attr_value) const
graphStatus SetAttrValue(const ge::DataType &attr_value) const
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| attr_value | 输入 | 具体的属性取值。 |

## 返回值说明

graphStatus 类型：成功，返回 `GRAPH_SUCCESS`，否则，返回 `GRAPH_FAILED`。

## 异常处理

无

## 约束说明

无
