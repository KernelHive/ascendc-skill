##### GetTilingKey

## 函数功能
获取 tiling key。

## 函数原型
```cpp
uint64_t GetTilingKey() const
```

## 参数说明
无

## 返回值说明
返回 tiling key。

## 约束说明
无

## 调用示例
```cpp
ge::graphStatus Tiling4XXX(TilingContext* context) {
    auto tiling_key = context->GetTilingKey();
    // ...
}
```
