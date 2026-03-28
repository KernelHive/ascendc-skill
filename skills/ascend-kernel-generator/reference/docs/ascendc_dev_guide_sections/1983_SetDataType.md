##### SetDataType

## 函数功能
设置 Tensor 的 Datatype。

## 函数原型
```cpp
graphStatus SetDataType(const ge::DataType &dtype)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| dtype  | 输入      | 需设置的 DataType 值。<br>关于 DataType 类型，请参见 DataType。 |

## 返回值
graphStatus 类型：设置成功返回 `GRAPH_SUCCESS`，否则，返回 `GRAPH_FAILED`。

## 异常处理
无。

## 约束说明
无。
