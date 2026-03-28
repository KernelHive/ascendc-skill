##### SetLocalMemorySize

## 函数功能
用于设置需要使用的 Local Memory 大小。不设置的情况下，默认为 0，即算子不需要使用 Local Memory。

该接口为预留接口，为后续功能做保留，不建议开发者使用，开发者无需关注。

## 函数原型
```cpp
ge::graphStatus SetLocalMemorySize(const uint32_t local_memory_size)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| local_memory_size | 输入 | Local Memory 大小 |

## 返回值说明
设置成功时返回 `ge::GRAPH_SUCCESS`。

关于 `graphStatus` 的定义，请参见 15.2.3.55 ge::graphStatus。

## 约束说明
无。

## 调用示例
```cpp
ge::graphStatus Tiling4XXX(TilingContext* context) {
    // ...
    auto ret = context->SetLocalMemorySize(1024 * 128);
}
```
