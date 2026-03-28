##### SetAttr

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

设置Node属性的属性值。

Node可以包括多个属性，初次设置值后，算子属性值的类型固定，算子属性值的类型包括：

- **整型**：接受int64_t、uint32_t、int32_t类型的整型值
  - 使用`SetAttr(const AscendString& name, int64_t &attrValue)`设置属性值
  - 以`GetAttr(const AscendString& name, int32_t &attrValue)`、`GetAttr(const AscendString& name, uint32_t &attrValue)`取值时，用户需保证整型数据没有截断，同理针对int32_t和uint32_t混用时需要保证不被截断
- **整型列表**：接受`std::vector<int64_t>`、`std::vector<int32_t>`、`std::vector<uint32_t>`表示的整型列表数据
- **浮点数**：float
- **浮点数列表**：`std::vector<float>`
- **字符串**：AscendString
- **字符串列表**：`std::vector<AscendString>`
- **布尔**：bool
- **布尔列表**：`std::vector<bool>`
- **Tensor**：Tensor
- **Tensor列表**：`std::vector<Tensor>`
- **OpBytes**：字节数组，`vector<uint8_t>`
- **AttrValue类型**
- **整型二维列表类型**：`std::vector<std::vector<int64_t>>`
- **DataType列表类型**：`vector<ge::DataType>`
- **DataType类型**：DataType

## 函数原型

```cpp
graphStatus SetAttr(const AscendString &name, int64_t &attr_value) const
graphStatus SetAttr(const AscendString &name, int32_t &attr_value) const
graphStatus SetAttr(const AscendString &name, uint32_t &attr_value) const
graphStatus SetAttr(const AscendString &name, float32_t &attr_value) const
graphStatus SetAttr(const AscendString &name, AscendString &attr_value) const
graphStatus SetAttr(const AscendString &name, bool &attr_value) const
graphStatus SetAttr(const AscendString &name, Tensor &attr_value) const
graphStatus SetAttr(const AscendString &name, std::vector<int64_t> &attr_value) const
graphStatus SetAttr(const AscendString &name, std::vector<int32_t> &attr_value) const
graphStatus SetAttr(const AscendString &name, std::vector<uint32_t> &attr_value) const
graphStatus SetAttr(const AscendString &name, std::vector<float32_t> &attr_value) const
graphStatus SetAttr(const AscendString &name, std::vector<AscendString> &attr_values) const
graphStatus SetAttr(const AscendString &name, std::vector<bool> &attr_value) const
graphStatus SetAttr(const AscendString &name, std::vector<Tensor> &attr_value) const
graphStatus SetAttr(const AscendString &name, OpBytes &attr_value) const
graphStatus SetAttr(const AscendString &name, std::vector<std::vector<int64_t>> &attr_value) const
graphStatus SetAttr(const AscendString &name, std::vector<ge::DataType> &attr_value) const
graphStatus SetAttr(const AscendString &name, ge::DataType &attr_value) const
graphStatus SetAttr(const AscendString &name, AttrValue &attr_value) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| name | 输入 | 属性名称 |
| attr_value | 输入 | 需设置的int64_t表示的整型类型属性值 |
| attr_value | 输入 | 需设置的int32_t表示的整型类型属性值 |
| attr_value | 输入 | 需设置的uint32_t表示的整型类型属性值 |
| attr_value | 输入 | 需设置的浮点类型float的属性值 |
| attr_value | 输入 | 需设置的字符串类型AscendString的属性值 |
| attr_value | 输入 | 需设置的布尔类型bool的属性值 |
| attr_value | 输入 | 需设置的Tensor类型的属性值 |
| attr_value | 输入 | 需设置的vector<int64_t>表示的整型列表类型属性值 |
| attr_value | 输入 | 需设置的vector<int32_t>表示的整型列表类型属性值 |
| attr_value | 输入 | 需设置的vector<uint32_t>表示的整型列表类型属性值 |
| attr_value | 输入 | 需设置的浮点列表类型vector<float>的属性值 |
| attr_value | 输入 | 需设置的字符串列表类型vector<ge::AscendString>的属性值 |
| attr_value | 输入 | 需设置的布尔列表类型vector<bool>的属性值 |
| attr_value | 输入 | 需设置的Tensor列表类型vector<Tensor>的属性值 |
| attr_value | 输入 | 需设置的Bytes，即字节数组类型的属性值，OpBytes即vector<uint8_t> |
| attr_value | 输入 | 需设置的vector<vector<int64_t>>表示的整型二维列表类型属性值 |
| attr_value | 输入 | 需设置的vector<ge::DataType>表示的DataType列表类型属性值 |
| attr_value | 输入 | 需设置的DataType类型的属性值 |
| attr_value | 输入 | 需设置的AttrValue类型的属性值 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| graphStatus | - | GRAPH_SUCCESS(0)：成功<br>其他值：失败 |

## 约束说明

无
