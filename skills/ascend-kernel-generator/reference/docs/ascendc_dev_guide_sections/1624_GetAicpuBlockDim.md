##### GetAicpuBlockDim

## 函数功能
获取用户设置的可以调度的AI CPU核数。设置方式请参考 [SetAicpuBlockDim](./15.2.2.35.15-SetAicpuBlockDim.md)。

## 函数原型
```cpp
uint32_t GetAicpuBlockDim() const
```

## 参数说明
无。

## 返回值说明
用户设置的可以调度的AI CPU核数。

## 约束说明
无。

## 调用示例
```cpp
ge::graphStatus Tiling4XXX(TilingContext* context) {
    auto block_dim = context->GetAicpuBlockDim();
    // ...
}
```
