##### ImplyType

## 函数功能

设置算子执行方式。

## 函数原型

```cpp
OpRegistrationData &ImplyType(const domi::ImplyType &imply_type)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| imply_type | 输入 | 算子执行方式 |

## ImplyType 枚举

```cpp
enum class ImplyType : unsigned int
{
    BUILDIN = 0,    // 内置算子，由 OME 正常执行
    TVM,            // 编译成 TVM bin 文件执行
    CUSTOM,         // 由用户自定义计算逻辑，通过 CPU 执行
    AI_CPU,         // AI CPU 自定义算子类型
    INVALID = 0xFFFFFFFF,
};
```
