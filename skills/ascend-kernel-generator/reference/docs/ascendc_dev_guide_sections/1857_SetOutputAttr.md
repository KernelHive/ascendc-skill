##### SetOutputAttr

## 函数功能

设置算子输出 Tensor 属性的属性值。

算子可以包括多个属性，初次设置值后，算子属性值的类型固定。算子属性值的类型包括：

- **整型**：接受 `int64_t`、`uint32_t`、`int32_t` 类型的整型值。
  以 `int64_t` 为例：
  - 使用：
    ```cpp
    SetOutputAttr(const char_t *dst_name, const char_t *name, int64_t attr_value);
    SetOutputAttr(const int32_t index, const char_t *name, int64_t attr_value);
    ```
  - 取值时：
    ```cpp
    GetOutputAttr(const int32_t index, const char_t *name, int64_t &attr_value) const;
    GetOutputAttr(const char_t *dst_name, const char_t *name, int64_t &attr_value) const;
    ```
  用户需保证整型数据没有截断，同理针对 `int32_t` 和 `uint32_t` 混用时需要保证不被截断。

- **整型列表**：接受 `std::vector<int64_t>`、`std::vector<int32_t>`、`std::vector<uint32_t>`、`std::initializer_list<int64_t>&&` 表示的整型列表数据。

- **浮点数**：`float32_t`

- **浮点数列表**：`std::vector<float32_t>`

- **字符串**：`string`

- **布尔**：`bool`

- **布尔列表**：`std::vector<bool>`

## 函数原型

```cpp
Operator &SetOutputAttr(const int32_t index, const char_t *name, const char_t *attr_value)
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const char_t *attr_value)
Operator &SetOutputAttr(const int32_t index, const char_t *name, const AscendString &attr_value)
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const AscendString &attr_value)
Operator &SetOutputAttr(const int32_t index, const char_t *name, int64_t attr_value)
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, int64_t attr_value)
Operator &SetOutputAttr(const int32_t index, const char_t *name, int32_t attr_value)
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, int32_t attr_value)
Operator &SetOutputAttr(const int32_t index, const char_t *name, uint32_t attr_value)
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, uint32_t attr_value)
Operator &SetOutputAttr(const int32_t index, const char_t *name, bool attr_value)
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, bool attr_value)
Operator &SetOutputAttr(const int32_t index, const char_t *name, float32_t attr_value)
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, float32_t attr_value)
Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<AscendString> &attr_value)
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<AscendString> &attr_value)
Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<int64_t> &attr_value)
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<int64_t> &attr_value)
Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<int32_t> &attr_value)
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<int32_t> &attr_value)
Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<uint32_t> &attr_value)
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<uint32_t> &attr_value)
Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<bool> &attr_value)
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<bool> &attr_value)
Operator &SetOutputAttr(const int32_t index, const char_t *name, const std::vector<float32_t> &attr_value)
Operator &SetOutputAttr(const char_t *dst_name, const char_t *name, const std::vector<float32_t> &attr_value)
```

## 参数说明

| 参数名     | 输入/输出 | 描述 |
|------------|-----------|------|
| name       | 输入      | 属性名称。 |
| index      | 输入      | 输出索引。 |
| dst_name   | 输入      | 输出边名称。 |
| attr_value | 输入      | 需设置的 `int64_t` 表示的整型类型属性值。 |
| attr_value | 输入      | 需设置的 `int32_t` 表示的整型类型属性值。 |
| attr_value | 输入      | 需设置的 `uint32_t` 表示的整型类型属性值。 |
| attr_value | 输入      | 需设置的 `vector<int64_t>` 表示的整型列表类型属性值。 |
| attr_value | 输入      | 需设置的 `vector<int32_t>` 表示的整型列表类型属性值。 |
| attr_value | 输入      | 需设置的 `vector<uint32_t>` 表示的整型列表类型属性值。 |
| attr_value | 输入      | 需设置的浮点类型的属性值。 |
| attr_value | 输入      | 需设置的浮点列表类型的属性值。 |
| attr_value | 输入      | 需设置的布尔类型的属性值。 |
| attr_value | 输入      | 需设置的布尔列表类型的属性值。 |
| attr_value | 输入      | 需设置的字符串类型的属性值。 |
| attr_value | 输入      | 需设置的字符串列表类型的属性值。 |

## 返回值

`Operator` 对象本身。

## 异常处理

无。

## 约束说明

无。
