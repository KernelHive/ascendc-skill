##### CreateOperator

## 函数功能

基于算子名称和算子类型获取算子对象实例。

## 函数原型

```cpp
static Operator CreateOperator(const std::string &operator_name, const std::string &operator_type)
static Operator CreateOperator(const char_t *const operator_name, const char_t *const operator_type)
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| operator_name | 输入 | 算子名称。 |
| operator_type | 输入 | 算子类型。 |

## 返回值

算子对象实例。

## 约束说明

无。
