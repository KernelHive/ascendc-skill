##### OpRegistrationData 构造函数和析构函数

## 函数功能

OpRegistrationData 构造函数和析构函数。

## 函数原型

```cpp
OpRegistrationData(const std::string &om_optype)
OpRegistrationData(const char_t *om_optype)
~OpRegistrationData()
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名     | 输入/输出 | 描述                               |
|------------|-----------|------------------------------------|
| om_optype  | 输入      | 指定适配昇腾 AI 处理器的模型支持的算子类型 |

## 返回值

OpRegistrationData 构造函数返回 OpRegistrationData 类型的对象。

## 异常处理

无。

## 约束说明

无。
