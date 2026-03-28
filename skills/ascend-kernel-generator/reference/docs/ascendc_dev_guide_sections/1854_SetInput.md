##### SetInput

## 函数功能

设置算子 Input，即由哪个算子的输出连接到本算子。

有以下几种 SetInput 方法：

- 如果指定 `srcOprt` 第 0 个 Output 为当前算子 Input，使用第一个函数原型设置当前算子 Input，不需要指定 `srcOprt` 的 Output 名称。
- 如果指定 `srcOprt` 的其它 Output 为当前算子 Input，使用第二个函数原型设置当前算子 Input，需要指定 `srcOprt` 的 Output 名称。
- 如果指定 `srcOprt` 的其它 Output 为当前算子 Input，使用第三个函数原型设置当前算子 Input，需要指定 `srcOprt` 的第 `index` 个 Output。

## 函数原型

```cpp
Operator &SetInput(const std::string &dst_name, const Operator &src_oprt)
Operator &SetInput(const char_t *dst_name, const Operator &src_oprt)

Operator &SetInput(const std::string &dst_name, const Operator &src_oprt, const std::string &name)
Operator &SetInput(const char_t *dst_name, const Operator &src_oprt, const char_t *name)
Operator &SetInput(const std::string &dst_name, const Operator &src_oprt, uint32_t index)
Operator &SetInput(const char_t *dst_name, const Operator &src_oprt, uint32_t index)
Operator &SetInput(uint32_t dst_index, const Operator &src_oprt, uint32_t src_index)
Operator &SetInput(const char_t *dst_name, uint32_t dst_index, const Operator &src_oprt, const char_t *name)
Operator &SetInput(const char_t *dst_name, uint32_t dst_index, const Operator &src_oprt)
```

## 须知

数据类型为 `string` 的接口后续版本会废弃，建议使用数据类型为非 `string` 的接口。

## 参数说明

| 参数名    | 输入/输出 | 描述 |
|-----------|-----------|------|
| dst_name  | 输入      | 当前算子 Input 名称 |
| src_oprt  | 输入      | Input 名称为 `dst_name` 的输入算子对象 |
| src_index | 输入      | `src_oprt` 的第 `src_index` 个输出 |
| name      | 输入      | `src_oprt` 的 Output 名称 |
| index     | 输入      | `src_oprt` 的第 `index` 个 Output |
| dst_index | 输入      | 名称为 `dst_name` 的第 `dst_index` 个动态输入 |

## 返回值

当前调度者本身。

## 异常处理

无。

## 约束说明

无。
