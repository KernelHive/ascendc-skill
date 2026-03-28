##### GetTilingCond

## 函数功能

获取 GetTilingCond 中设置的 tiling cond。

## 函数原型

```cpp
int32_t GetTilingCond() const
```

## 参数说明

无。

## 返回值说明

- 若返回值大于等于 0，代表此 tiling cond 为有效的 tiling cond。
- 若返回值为 -1，代表此 tiling cond 为无效的 tiling cond。

## 约束说明

无。

## 调用示例

```cpp
ge::graphStatus Tiling4XXX(TilingContext* context) {
  auto tiling_cond = context->GetTilingCond();
  // ...
}
```
