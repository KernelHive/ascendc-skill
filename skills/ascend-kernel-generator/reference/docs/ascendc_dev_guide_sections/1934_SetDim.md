##### SetDim

## 函数功能

将 Shape 中第 idx 维度的值设置为 value。

## 函数原型

```cpp
graphStatus SetDim(size_t idx, int64_t value)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| idx    | 输入      | Shape 维度的索引，索引从 0 开始。 |
| value  | 输入      | 需设置的值。 |

## 返回值

graphStatus 类型：设置成功返回 `GRAPH_SUCCESS`，否则返回 `GRAPH_FAILED`。

## 异常处理

无。

## 约束说明

使用 SetDim 接口前，只能使用 `Shape(const std::vector<int64_t>& dims)` 构造 Shape 对象。如果使用 `Shape()` 构造 Shape 对象，使用 SetDim 接口将返回失败。
