##### IsExistOp

## 函数功能

查询指定的算子类型是否支持。

## 函数原型

```cpp
static bool IsExistOp(const std::string &operator_type)
static bool IsExistOp(const char_t *const operator_type)
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| operator_type | 输入 | 算子类型 |

## 返回值

- `true`：存在此算子
- `false`：不存在此算子

## 约束说明

无。
