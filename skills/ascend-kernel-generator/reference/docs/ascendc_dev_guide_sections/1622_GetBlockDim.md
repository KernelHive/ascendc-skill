##### GetBlockDim

## 函数功能
获取 blockDim，即参与计算的 Vector 或者 Cube 核数。blockDim 的详细概念和设置方式请参考 15.2.2.35.13 SetBlockDim。

## 函数原型
```cpp
uint32_t GetBlockDim() const
```

## 参数说明
无。

## 返回值说明
返回 blockDim。

## 约束说明
无。

## 调用示例
```cpp
ge::graphStatus Tiling4XXX(TilingContext* context) {
    auto block_dim = context->GetBlockDim();
    // ...
}
```
