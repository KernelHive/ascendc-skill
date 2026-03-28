##### SetOriginShapeDimNum

## 函数功能
设置原始 shape 的维度大小，即 rank 大小。

## 函数原型
```cpp
graphStatus SetOriginShapeDimNum(const size_t dim_num)
```

## 参数说明

| 参数名   | 输入/输出 | 描述                               |
|----------|-----------|------------------------------------|
| dim_num  | 输入      | 原始 shape 的维度大小，即原始 shape 的 rank。 |

## 返回值
graphStatus 类型：设置成功返回 `GRAPH_SUCCESS`，否则返回 `GRAPH_FAILED`。

## 异常处理
无。

## 约束说明
无。
