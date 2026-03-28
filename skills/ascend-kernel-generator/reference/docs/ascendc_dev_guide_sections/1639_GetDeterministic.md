##### GetDeterministic

## 函数功能
获取确定性计算配置选项。

## 函数原型
```cpp
int32_t GetDeterministic() const
```

## 参数说明
无。

## 返回值说明
- `0`：未开启确定性配置选项
- `1`：开启确定性配置选项

## 约束说明
无。

## 调用示例
```cpp
ge::graphStatus Tiling4XXX(TilingContext* context) {
    int32_t ret = context->GetDeterministic();
    // ...
}
```
