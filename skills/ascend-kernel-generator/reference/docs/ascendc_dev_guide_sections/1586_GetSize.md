##### GetSize

## 函数功能
获取 Tensor 数据的内存大小。

## 函数原型
```cpp
size_t GetSize() const
```

## 参数说明
无。

## 返回值说明
内存大小，单位是字节。

## 约束说明
无。

## 调用示例
```cpp
StorageShape sh({1, 2, 3}, {1, 2, 3});
Tensor t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
auto td_size = t.GetSize(); // 1*2*3*sizeof(float) = 24
```
