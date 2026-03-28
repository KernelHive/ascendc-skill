##### GetAttr

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

获取指定属性名字的值。

## 函数原型

```cpp
graphStatus GetAttr(const AscendString &name, int64_t &attr_value) const
graphStatus GetAttr(const AscendString &name, int32_t &attr_value) const
graphStatus GetAttr(const AscendString &name, uint32_t &attr_value) const
graphStatus GetAttr(const AscendString &name, float32_t &attr_value) const
graphStatus GetAttr(const AscendString &name, AscendString &attr_value) const
graphStatus GetAttr(const AscendString &name, bool &attr_value) const
graphStatus GetAttr(const AscendString &name, Tensor &attr_value) const
graphStatus GetAttr(const AscendString &name, std::vector<int64_t> &attr_value) const
graphStatus GetAttr(const AscendString &name, std::vector<int32_t> &attr_value) const
graphStatus GetAttr(const AscendString &name, std::vector<uint32_t> &attr_value) const
graphStatus GetAttr(const AscendString &name, std::vector<float32_t> &attr_value) const
graphStatus GetAttr(const AscendString &name, std::vector<AscendString> &attr_values) const
graphStatus GetAttr(const AscendString &name, std::vector<bool> &attr_value) const
graphStatus GetAttr(const AscendString &name, std::vector<Tensor> &attr_value) const
graphStatus GetAttr(const AscendString &name, OpBytes &attr_value) const
graphStatus GetAttr(const AscendString &name, std::vector<std::vector<int64_t>> &attr_value) const
graphStatus GetAttr(const AscendString &name, std::vector<ge::DataType> &attr_value) const
graphStatus GetAttr(const AscendString &name, ge::DataType &attr_value) const
graphStatus GetAttr(const AscendString &name, AttrValue &attr_value) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| name | 输入 | 属性名称。 |
| attr_value | 输出 | 返回的int64_t表示的整型类型属性值。 |
| attr_value | 输出 | 返回的int32_t表示的整型类型属性值。 |
| attr_value | 输出 | 返回的uint32_t表示的整型类型属性值。 |
| attr_value | 输出 | 返回的浮点类型float的属性值。 |
| attr_value | 输出 | 返回的字符串类型AscendString的属性值。 |
| attr_value | 输出 | 返回的布尔类型bool的属性值。 |
| attr_value | 输出 | 返回的Tensor类型的属性值。 |
| attr_value | 输出 | 返回的vector<int64_t>表示的整型列表类型属性值。 |
| attr_value | 输出 | 返回的vector<int32_t>表示的整型列表类型属性值。 |
| attr_value | 输出 | 返回的vector<uint32_t>表示的整型列表类型属性值。 |
| attr_value | 输出 | 返回的浮点列表类型vector<float>的属性值。 |
| attr_value | 输出 | 返回的字符串列表类型vector<ge::AscendString>的属性值。 |
| attr_value | 输出 | 返回的布尔列表类型vector<bool>的属性值。 |
| attr_value | 输出 | 返回的Tensor列表类型vector<Tensor>的属性值。 |
| attr_value | 输出 | 返回的Bytes，即字节数组类型的属性值，OpBytes即vector<uint8_t>。 |
| attr_value | 输出 | 返回的vector<vector<int64_t>>表示的整型二维列表类型属性值。 |
| attr_value | 输出 | 返回的vector<ge::DataType>表示的DataType列表类型属性值。 |
| attr_value | 输出 | 返回的DataType类型的属性值。 |
| attr_value | 输出 | 返回的AttrValue类型的属性值。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | graphStatus | GRAPH_SUCCESS(0)：成功。<br>其他值：失败。 |

## 约束说明

无
