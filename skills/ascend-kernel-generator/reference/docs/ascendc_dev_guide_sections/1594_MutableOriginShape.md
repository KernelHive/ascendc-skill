##### MutableOriginShape

## 函数功能
获取 Tensor 的原始 shape。

## 函数原型
```cpp
Shape &MutableOriginShape()
```

## 参数说明
无。

## 返回值说明
原始 shape 引用。

关于 Shape 类型的定义，请参见 15.2.2.27 Shape。

## 约束说明
无。

## 调用示例
```cpp
StorageShape sh({1, 2, 3}, {2, 1, 3});
Tensor t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
auto shape = t.MutableOriginShape(); // 1,2,3
```
