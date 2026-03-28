##### GetFormat

## 函数功能
获取 Tensor 的 format，包含运行时 format 和原始 format，返回的 format 对象都是只读的。

## 函数原型
```cpp
const StorageFormat &GetFormat() const
```

## 参数说明
无。

## 返回值说明
只读的 format 引用。

关于 StorageFormat 类型的定义，请参见 15.2.2.28 StorageFormat。

## 约束说明
无。

## 调用示例
```cpp
Tensor tensor{{{8, 3, 224, 224}, {16, 3, 224, 224}}, // shape
              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, // format
              kFollowing, // placement
              ge::DT_FLOAT16, // dt
              nullptr};
auto fmt = tensor.GetFormat();
```
