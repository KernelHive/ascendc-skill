##### MutableStorageShape

## 函数功能
获取运行时Tensor的shape，此shape对象是可变的。

## 函数原型
```cpp
Shape &MutableStorageShape()
```

## 参数说明
无。

## 返回值说明
运行时shape的引用。

## 约束说明
无。

## 调用示例
```cpp
StorageShape sh({1, 2, 3}, {2, 1, 3});
Tensor t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
auto shape = t.MutableStorageShape(); // 2,1,3
```
