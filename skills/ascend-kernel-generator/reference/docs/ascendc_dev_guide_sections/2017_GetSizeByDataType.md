#### GetSizeByDataType

## 函数功能

根据传入的 `data_type`，获取该数据类型所占用的内存大小。如果要获取多个元素的内存大小，请使用 `GetSizeInBytes`。

## 函数原型

```cpp
inline int GetSizeByDataType(DataType data_type)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| `data_type` | 输入 | 数据类型，请参见 15.2.3.58 DataType。 |

## 返回值

该 `data_type` 所占用的内存大小（单位为 bytes）。

- 如果该 `data_type` 所占用的内存小于 1 byte，返回 `1000 + 该数据类型的 bit 位数`，例如 `DT_INT4` 数据类型，返回 1004。
- 如果传入非法值或不支持的数据类型，返回 -1。

## 异常处理

无。

## 约束说明

无。
