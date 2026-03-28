##### CreateFrom

## 函数功能

将传入的 DT 类型（支持 `int64_t`、`float`、`std::string` 类型）的参数转换为对应 T 类型（支持 `INT`、`FLOAT`、`STR` 类型）的参数。

- 支持将 `int64_t` 类型转换为 `INT` 类型
- 支持将 `float` 类型转换为 `FLOAT` 类型
- 支持将 `std::string` 类型转换为 `STR` 类型

## 函数原型

```cpp
template<typename T, typename DT>
static T CreateFrom(DT &&val)
```

## 参数说明

| 参数名 | 输入/输出 | 描述             |
|--------|-----------|------------------|
| val    | 输入      | DT 类型的参数。 |

## 返回值

返回 T 类型的参数。

## 异常处理

无。

## 约束说明

无。
