##### GetPlacement

## 函数功能
获取 tensor 的 placement。

## 函数原型
```cpp
TensorPlacement GetPlacement() const
```

## 参数说明
无。

## 返回值说明
返回 tensor 的 placement。

关于 TensorPlacement 类型的定义，请参见 [15.2.2.38 TensorPlacement](#152238-tensorplacement)。

## 约束说明
无。

## 调用示例
```cpp
Tensor tensor{
    {{8, 3, 224, 224}, {16, 3, 224, 224}}, // shape
    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, // format
    kFollowing, // placement
    ge::DT_FLOAT16, // dt
    nullptr
};
auto placement = tensor.GetPlacement(); // kFollowing
```
