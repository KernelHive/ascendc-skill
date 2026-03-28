##### GetStorageShape

## 函数功能
获取运行时 Tensor 的 StorageShape，此 shape 对象为只读。StorageShape 和 Originshape 的区别如下：Originshape 是 Tensor 最初创建时的形状，StorageShape 是保存 Tensor 数据的底层存储的形状。运行时为了适配底层硬件，Tensor 的 StorageShape 和其 Originshape 可能会有所不同。

## 函数原型
```cpp
const Shape &GetStorageShape() const
```

## 参数说明
无。

## 返回值说明
只读的运行时 shape 引用。

## 约束说明
无。

## 调用示例
```cpp
StorageShape sh({1, 2, 3}, {2, 1, 3});
Tensor t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
auto shape = t.GetStorageShape(); // 2,1,3
```
