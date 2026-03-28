##### OperatorCreatorRegister 构造函数和析构函数

## 函数功能

OperatorCreatorRegister 构造函数和析构函数。

## 函数原型

```cpp
OperatorCreatorRegister(const std::string &operator_type, OpCreator const &op_creator)
OperatorCreatorRegister(const char_t *const operator_type, OpCreatorV2 const &op_creator)
~OperatorCreatorRegister() = default
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| operator_type | 输入 | 算子类型 |
| op_creator | 输入 | 算子构造函数 |

## 返回值

OperatorCreatorRegister 构造函数返回 OperatorCreatorRegister 类型的对象。

## 约束说明

算子注册接口，注册一个算子原型，此接口被其他头文件引用，一般不用由算子开发者直接调用。
