##### GetRawTilingData

## 函数功能
获取无类型的 tiling data 指针。

## 函数原型
```cpp
TilingData *GetRawTilingData()
```

## 参数说明
无。

## 返回值说明
tiling data 指针，失败时返回空指针。

## 约束说明
无。

## 调用示例
```cpp
ge::graphStatus Tiling4XXX(TilingContext* context) {
    auto tiling_data = context->GetRawTilingData();
    // ...
}
```
