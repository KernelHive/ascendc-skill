##### GetTilingData

## 函数功能
获取有类型的 tiling data 指针。

## 函数原型
```cpp
template<typename T> auto GetTilingData() -> T*
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `T`  | tiling data 类型，`sizeof(T)` 不可以大于编译结果中指定的最大 tiling data 长度。 |

## 返回值说明
tiling data 指针，失败时返回空指针。

## 约束说明
`sizeof(T)` 不可以大于编译结果中指定的最大 tiling data 长度。

## 调用示例
```cpp
ge::graphStatus Tiling4XXX(TilingContext* context) {
    auto tiling_data = context->GetTilingData<int64_t>();
    // ...
}
```
