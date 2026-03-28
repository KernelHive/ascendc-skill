##### TilingDataSize

## 函数功能

设置算子 TilingData 的大小。设置该大小后，会申请相应大小的内存用于存储算子的 TilingData。相较于 TilingData 接口，调用此接口时生成的 TilingData 指针所有权归属 ContextHolder，调用者无需关注 TilingData 的生命周期。

> **注意**：该接口与 `TilingData` 互斥。如果同时调用 `TilingDataSize` 和 `TilingData` 接口，后调用的接口会覆盖前一次调用的结果，以最新的为准。

## 函数原型

```cpp
OpTilingContextBuilder &TilingDataSize(size_t tiling_data_size)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| `tiling_data_size` | 输出 | Tiling 数据大小 |

## 返回值说明

返回 `TilingContextBuilder` 对象，用于链式调用。

## 约束说明

在调用 `Build` 方法之前，必须设置 `TilingData` 或 `TilingDataSize`，否则构造出的 `TilingContext` 将包含未定义数据。
