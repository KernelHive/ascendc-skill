##### SetTensorDesc

## 函数功能

设置 Tensor 的描述符（TensorDesc）。

## 函数原型

```cpp
graphStatus SetTensorDesc(const TensorDesc &tensor_desc)
```

## 参数说明

| 参数名       | 输入/输出 | 描述                   |
|--------------|-----------|------------------------|
| tensor_desc  | 输入      | 需设置的 Tensor 描述符 |

## 返回值

graphStatus 类型：设置成功返回 `GRAPH_SUCCESS`，否则返回 `GRAPH_FAILED`。

## 异常处理

无。

## 约束说明

无。
