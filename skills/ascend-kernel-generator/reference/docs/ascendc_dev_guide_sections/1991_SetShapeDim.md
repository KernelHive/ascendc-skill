##### SetShapeDim

## 函数功能
设置 shape 第 idx 维度。

## 函数原型
```cpp
graphStatus SetShapeDim(const size_t idx, const int64_t dim_value)
```

## 参数说明

| 参数名     | 输入/输出 | 描述                         |
|------------|-----------|------------------------------|
| idx        | 输入      | 维度的索引，索引从 0 开始。  |
| dim_value  | 输入      | 需设置的值。                 |

## 返回值
graphStatus 类型：设置成功返回 `GRAPH_SUCCESS`，否则，返回 `GRAPH_FAILED`。

## 异常处理
无。

## 约束说明
无。
