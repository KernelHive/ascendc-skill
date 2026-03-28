##### GetAddr

## 函数功能
获取Tensor的数据地址。

## 函数原型
```cpp
const void *GetAddr() const
void *GetAddr()
```

## 参数说明
无。

## 返回值说明
返回数据地址。

## 约束说明
无。

## 调用示例
```cpp
Tensor tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}}, // shape
{ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, // format
kFollowing, // placement
ge::DT_FLOAT16, // dt
nullptr};
auto addr = tensor.GetAddr(); // &tensor + 1
```
