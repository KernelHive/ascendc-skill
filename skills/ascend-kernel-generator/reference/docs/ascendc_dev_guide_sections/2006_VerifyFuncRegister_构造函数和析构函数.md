##### VerifyFuncRegister 构造函数和析构函数

## 函数功能

VerifyFuncRegister 构造函数和析构函数。

## 函数原型

```cpp
VerifyFuncRegister(const std::string &operator_type, const VerifyFunc &verify_func)
VerifyFuncRegister(const char_t *const operator_type, const VerifyFunc &verify_func)
~VerifyFuncRegister() = default
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| operator_type | 输入 | 算子类型 |
| verify_func | 输入 | 算子 verify 函数 |

## 返回值

VerifyFuncRegister 构造函数返回 VerifyFuncRegister 类型的对象。

## 约束说明

算子 verifyFunc 函数注册接口，此接口被其他头文件引用，一般不用由算子开发者直接调用。
