##### GetShapeSize

## 函数功能

获取当前 Tensor 运行时的 shape 大小，即此 Tensor 中包含的元素的数量。

## 函数原型

```cpp
int64_t GetShapeSize() const
```

## 参数说明

无。

## 返回值说明

返回执行时 shape 的大小。

## 约束说明

无。

## 调用示例

```cpp
Tensor tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}}, // shape
{ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, // format
kFollowing, // placement
ge::DT_FLOAT16, // dt
nullptr};
auto shape_size = tensor.GetShapeSize(); // 16*3*224*224
```
