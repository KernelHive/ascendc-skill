#### GetSizeInBytes

## 函数功能

根据传入的 `element_count` 和 `data_type`，获取 `element_count` 个该 `data_type` 所占用的内存总大小。

## 函数原型

```cpp
int64_t GetSizeInBytes(int64_t element_count, DataType data_type)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| `element_count` | 输入 | 用于标识多少个 `data_type` |
| `data_type` | 输入 | 数据类型，请参见 15.2.3.58 DataType |

## 返回值

如果传入个数为非法值或传入不支持的数据类型，返回 `-1`。

## 异常处理

无。

## 约束说明

无。
