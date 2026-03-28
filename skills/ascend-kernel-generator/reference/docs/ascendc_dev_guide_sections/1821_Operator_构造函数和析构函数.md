##### Operator 构造函数和析构函数

## 函数功能
Operator 构造函数和析构函数。

## 函数原型

```cpp
Operator()
explicit Operator(const std::string &type)
explicit Operator(const char_t *type)
Operator(const std::string &name, const std::string &type)
Operator(const AscendString &name, const AscendString &type)
Operator(const char_t *name, const char_t *type)
virtual ~Operator() = default
```

## 须知
数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名 | 输入/输出 | 描述       |
|--------|-----------|------------|
| type   | 输入      | 算子类型。 |
| name   | 输入      | 算子名称。 |

## 返回值
Operator 构造函数返回 Operator 类型的对象。

## 异常处理
无。

## 约束说明
无。
