##### GetAttr

## 函数功能

根据属性名称获取对应的属性值。

## 函数原型

```cpp
graphStatus GetAttr(const std::string &name, int64_t &attr_value) const
graphStatus GetAttr(const std::string &name, int32_t &attr_value) const
graphStatus GetAttr(const std::string &name, uint32_t &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<int64_t> &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<int32_t> &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<uint32_t> &attr_value) const
graphStatus GetAttr(const std::string &name, float32_t &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<float32_t> &attr_value) const
graphStatus GetAttr(const std::string &name, AttrValue &attr_value) const
graphStatus GetAttr(const std::string &name, std::string &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<std::string> &attr_value) const
graphStatus GetAttr(const std::string &name, bool &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<bool> &attr_value) const
graphStatus GetAttr(const std::string &name, Tensor &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<Tensor> &attr_value) const
graphStatus GetAttr(const std::string &name, OpBytes &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<std::vector<int64_t>> &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<ge::DataType> &attr_value) const
graphStatus GetAttr(const std::string &name, ge::DataType &attr_value) const
graphStatus GetAttr(const std::string &name, ge::NamedAttrs &attr_value) const
graphStatus GetAttr(const std::string &name, std::vector<ge::NamedAttrs> &attr_value) const

graphStatus GetAttr(const char_t *name, int64_t &attr_value) const
graphStatus GetAttr(const char_t *name, int32_t &attr_value) const
graphStatus GetAttr(const char_t *name, uint32_t &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<int64_t> &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<int32_t> &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<uint32_t> &attr_value) const
graphStatus GetAttr(const char_t *name, float32_t &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<float32_t> &attr_value) const
graphStatus GetAttr(const char_t *name, AttrValue &attr_value) const
graphStatus GetAttr(const char_t *name, AscendString &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<AscendString> &attr_values) const
graphStatus GetAttr(const char_t *name, bool &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<bool> &attr_value) const
graphStatus GetAttr(const char_t *name, Tensor &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<Tensor> &attr_value) const
graphStatus GetAttr(const char_t *name, OpBytes &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<std::vector<int64_t>> &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<ge::DataType> &attr_value) const
graphStatus GetAttr(const char_t *name, ge::DataType &attr_value) const
graphStatus GetAttr(const char_t *name, ge::NamedAttrs &attr_value) const
graphStatus GetAttr(const char_t *name, std::vector<ge::NamedAttrs> &attr_value) const
```

## 须知

数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| name | 输入 | 属性名称。 |
| attr_value | 输出 | 返回的int64_t表示的整型类型属性值。 |
| attr_value | 输出 | 返回的int32_t表示的整型类型属性值。 |
| attr_value | 输出 | 返回的uint32_t表示的整型类型属性值。 |
| attr_value | 输出 | 返回的vector<int64_t>表示的整型列表类型属性值。 |
| attr_value | 输出 | 返回的vector<int32_t>表示的整型列表类型属性值。 |
| attr_value | 输出 | 返回的vector<uint32_t>表示的整型列表类型属性值。 |
| attr_value | 输出 | 返回的浮点类型的属性值。 |
| attr_value | 输出 | 返回的浮点列表类型的属性值。 |
| attr_value | 输出 | 返回的AttrValue类型的属性值。 |
| attr_value | 输出 | 返回的布尔类型的属性值。 |
| attr_value | 输出 | 返回的布尔列表类型的属性值。 |
| attr_value | 输出 | 返回的字符串类型的属性值。 |
| attr_value | 输出 | 返回的字符串列表类型的属性值。 |
| attr_value | 输出 | 返回的Tensor类型的属性值。 |
| attr_value | 输出 | 返回的Tensor列表类型的属性值。 |
| attr_value | 输出 | 返回的Bytes，即字节数组类型的属性值，OpBytes即vector<uint8_t>。 |
| attr_value | 输出 | 返回的量化数据的属性值。 |
| attr_value | 输出 | 返回的vector<vector<int64_t>>表示的整型二维列表类型属性值。 |
| attr_value | 输出 | 返回的vector<ge::DataType>表示的DataType列表类型属性值。 |
| attr_value | 输出 | 返回的DataType类型的属性值。 |
| attr_value | 输出 | 返回的vector<ge::NamedAttrs>表示的NamedAttrs列表类型属性值。 |
| attr_value | 输出 | 返回的NamedAttrs类型的属性值。 |

## 返回值

graphStatus类型：找到对应name，返回GRAPH_SUCCESS，否则返回GRAPH_FAILED。

## 异常处理

无。

## 约束说明

无。
