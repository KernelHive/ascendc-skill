##### SetAttr

## 函数功能
设置算子属性的属性值。

算子可以包括多个属性，初次设置值后，算子属性值的类型固定。算子属性值的类型包括：

- **整型**：接受 `int64_t`、`uint32_t`、`int32_t` 类型的整型值
  - 使用 `SetAttr(const string& name, int64_t attrValue)` 设置属性值
  - 使用 `GetAttr(const string& name, int32_t& attrValue)`、`GetAttr(const string& name, uint32_t& attrValue)` 取值时，用户需保证整型数据没有截断
  - 同理针对 `int32_t` 和 `uint32_t` 混用时需要保证不被截断
- **整型列表**：接受 `std::vector<int64_t>`、`std::vector<int32_t>`、`std::vector<uint32_t>`、`std::initializer_list<int64_t>&&` 表示的整型列表数据
- **浮点数**：`float`
- **浮点数列表**：`std::vector<float>`
- **字符串**：`string`
- **字符串列表**：`std::vector<std::string>`
- **布尔**：`bool`
- **布尔列表**：`std::vector<bool>`
- **Tensor**：`Tensor`
- **Tensor列表**：`std::vector<Tensor>`
- **Bytes**：字节数组，`SetAttr` 接受通过 `OpBytes`（即 `vector<uint8_t>`）和 `(const uint8_t* data, size_t size)` 表示的字节数组
- **AttrValue类型**
- **整型二维列表类型**：`std::vector<std::vector<int64_t>>`
- **DataType列表类型**：`std::vector<ge::DataType>`
- **DataType类型**：`ge::DataType`
- **NamedAttrs类型**：`ge::NamedAttrs`
- **NamedAttrs列表类型**：`std::vector<ge::NamedAttrs>`

## 函数原型

> **须知**
>
> 数据类型为 `string` 的接口后续版本会废弃，建议使用数据类型为非 `string` 的接口。

```cpp
Operator &SetAttr(const std::string &name, int64_t attr_value)
Operator &SetAttr(const std::string &name, int32_t attr_value)
Operator &SetAttr(const std::string &name, uint32_t attr_value)
Operator &SetAttr(const std::string &name, const std::vector<int64_t> &attr_value)
Operator &SetAttr(const std::string &name, const std::vector<int32_t> &attr_value)
Operator &SetAttr(const std::string &name, const std::vector<uint32_t> &attr_value)
Operator &SetAttr(const std::string &name, std::initializer_list<int64_t> &&attr_value)
Operator &SetAttr(const std::string &name, float32_t attr_value)
Operator &SetAttr(const std::string &name, const std::vector<float32_t> &attr_value)
Operator &SetAttr(const std::string &name, AttrValue &&attr_value)
Operator &SetAttr(const std::string &name, const std::string &attr_value)
Operator &SetAttr(const std::string &name, const std::vector<std::string> &attr_value)
Operator &SetAttr(const std::string &name, bool attr_value)
Operator &SetAttr(const std::string &name, const std::vector<bool> &attr_value)
Operator &SetAttr(const std::string &name, const Tensor &attr_value)
Operator &SetAttr(const std::string &name, const std::vector<Tensor> &attr_value)
Operator &SetAttr(const std::string &name, const OpBytes &attr_value)
Operator &SetAttr(const std::string &name, const std::vector<std::vector<int64_t>> &attr_value)
Operator &SetAttr(const std::string &name, const std::vector<ge::DataType> &attr_value)
Operator &SetAttr(const std::string &name, const ge::DataType &attr_value)
Operator &SetAttr(const std::string &name, const ge::NamedAttrs &attr_value)
Operator &SetAttr(const std::string &name, const std::vector<ge::NamedAttrs> &attr_value)

Operator &SetAttr(const char_t *name, int64_t attr_value)
Operator &SetAttr(const char_t *name, int32_t attr_value)
Operator &SetAttr(const char_t *name, uint32_t attr_value)
Operator &SetAttr(const char_t *name, const std::vector<int64_t> &attr_value)
Operator &SetAttr(const char_t *name, const std::vector<int32_t> &attr_value)
Operator &SetAttr(const char_t *name, const std::vector<uint32_t> &attr_value)
Operator &SetAttr(const char_t *name, std::initializer_list<int64_t> &&attr_value)
Operator &SetAttr(const char_t *name, float32_t attr_value)
Operator &SetAttr(const char_t *name, const std::vector<float32_t> &attr_value)
Operator &SetAttr(const char_t *name, AttrValue &&attr_value)
Operator &SetAttr(const char_t *name, const char_t *attr_value)
Operator &SetAttr(const char_t *name, const AscendString &attr_value)
Operator &SetAttr(const char_t *name, const std::vector<AscendString> &attr_values)
Operator &SetAttr(const char_t *name, bool attr_value)
Operator &SetAttr(const char_t *name, const std::vector<bool> &attr_value)
Operator &SetAttr(const char_t *name, const Tensor &attr_value)
Operator &SetAttr(const char_t *name, const std::vector<Tensor> &attr_value)
Operator &SetAttr(const char_t *name, const OpBytes &attr_value)
Operator &SetAttr(const char_t *name, const std::vector<std::vector<int64_t>> &attr_value)
Operator &SetAttr(const char_t *name, const std::vector<ge::DataType> &attr_value)
Operator &SetAttr(const char_t *name, const ge::DataType &attr_value)
Operator &SetAttr(const char_t *name, const ge::NamedAttrs &attr_value)
Operator &SetAttr(const char_t *name, const std::vector<ge::NamedAttrs> &attr_value)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| name | 输入 | 属性名称 |
| attr_value | 输入 | 需设置的 `int64_t` 表示的整型类型属性值 |
| attr_value | 输入 | 需设置的 `int32_t` 表示的整型类型属性值 |
| attr_value | 输入 | 需设置的 `uint32_t` 表示的整型类型属性值 |
| attr_value | 输入 | 需设置的 `vector<int64_t>` 表示的整型列表类型属性值 |
| attr_value | 输入 | 需设置的 `vector<int32_t>` 表示的整型列表类型属性值 |
| attr_value | 输入 | 需设置的 `vector<uint32_t>` 表示的整型列表类型属性值 |
| attr_value | 输入 | 需设置的 `std::initializer_list<int64_t>&&` 表示的整型列表类型属性值 |
| attr_value | 输入 | 需设置的浮点类型的属性值 |
| attr_value | 输入 | 需设置的浮点列表类型的属性值 |
| attr_value | 输入 | 需设置的布尔类型的属性值 |
| attr_value | 输入 | 需设置的布尔列表类型的属性值 |
| attr_value | 输入 | 需设置的 `AttrValue` 类型的属性值 |
| attr_value | 输入 | 需设置的字符串类型的属性值 |
| attr_value | 输入 | 需设置的字符串列表类型的属性值 |
| attr_value | 输入 | 需设置的 `Tensor` 类型的属性值 |
| attr_value | 输入 | 需设置的 `Tensor` 列表类型的属性值 |
| attr_value | 输入 | 需设置的 `Bytes`，即字节数组类型的属性值，`OpBytes` 即 `vector<uint8_t>` |
| data | 输入 | 需设置的 `Bytes`，即字节数组类型的属性值，指定了字节流的首地址 |
| size | 输入 | 需设置的 `Bytes`，即字节数组类型的属性值，指定了字节流的长度 |
| attr_value | 输入 | 需设置的量化数据的属性值 |
| attr_value | 输入 | 需设置的 `vector<vector<int64_t>>` 表示的整型二维列表类型属性值 |
| attr_value | 输入 | 需设置的 `vector<ge::DataType>` 表示的 `DataType` 列表类型属性值 |
| attr_value | 输入 | 需设置的 `DataType` 类型的属性值 |
| attr_value | 输入 | 需设置的 `NamedAttrs` 类型的属性值 |
| attr_value | 输入 | 需设置的 `vector<ge::NamedAttrs>` 表示的 `NamedAttrs` 列表类型的属性值 |

## 返回值
`Operator` 对象本身。

## 异常处理
无。

## 约束说明
无。
