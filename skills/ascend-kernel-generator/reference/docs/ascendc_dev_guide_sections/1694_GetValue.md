##### GetValue

## 函数功能

获取属性 key-value 键值对中的 value 值，并将 value 值从 T 类型转换为 DT 类型。

- 支持将 INT 类型转换为 int64_t 类型
- 支持将 FLOAT 类型转换为 float 类型
- 支持将 STR 类型转换为 std::string 类型

## 函数原型

```cpp
template<typename T, typename DT>
graphStatus GetValue(DT &val) const

graphStatus GetValue(AscendString &val)
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| val    | 输出      | DT 类型的参数 |

## 返回值

graphStatus 类型：数据类型转换成功，返回 `GRAPH_SUCCESS`，否则，返回 `GRAPH_FAILED`。

## 异常处理

无。

## 约束说明

无。
