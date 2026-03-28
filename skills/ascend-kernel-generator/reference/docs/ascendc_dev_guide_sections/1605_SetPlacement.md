##### SetPlacement

## 函数功能
设置 tensor 的 placement。

## 函数原型
```cpp
void SetPlacement(const TensorPlacement placement)
```

## 参数说明
| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| placement | 输入 | 需要设置的 tensor 的 placement。<br>关于 TensorPlacement 类型的定义，请参见 15.2.2.38 TensorPlacement。 |

## 返回值说明
无。

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
tensor.SetPlacement(TensorPlacement::kOnHost);
auto placement = tensor.GetPlacement();
```
