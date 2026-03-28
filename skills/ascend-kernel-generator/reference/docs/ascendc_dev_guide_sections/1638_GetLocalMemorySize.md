##### GetLocalMemorySize

## 函数功能
算子获取所需的 Local Memory 大小。

该接口为预留接口，为后续功能做保留，不建议开发者使用，开发者无需关注。

## 函数原型
```cpp
uint32_t GetLocalMemorySize()
```

## 参数说明
无。

## 返回值说明
返回 Local Memory 大小，如果之前没有调用 `SetLocalMemorySize` 进行设置，则返回 0。

## 约束说明
无。

## 调用示例
```cpp
ge::graphStatus Tiling4XXX(TilingContext* context) {
    auto local_memory_size = context->GetLocalMemorySize();
    // ...
}
```
