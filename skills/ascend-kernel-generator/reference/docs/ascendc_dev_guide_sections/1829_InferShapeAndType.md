##### InferShapeAndType

## 函数功能

推导 Operator 输出的 shape 和 DataType。

关于 DataType 数据类型的定义，请参见 15.2.3.58 DataType。

## 函数原型

```cpp
graphStatus InferShapeAndType()
```

## 参数说明

无。

## 返回值

graphStatus 类型：推导成功，返回 `GRAPH_SUCCESS`，否则，返回 `GRAPH_FAILED`。

## 异常处理

无。

## 约束说明

无。
