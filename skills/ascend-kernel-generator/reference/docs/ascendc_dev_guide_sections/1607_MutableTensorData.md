##### MutableTensorData

## 函数功能
获取 tensor 中的数据。

## 函数原型
```cpp
TensorData &MutableTensorData()
```

## 参数说明
无。

## 返回值说明
可写的 tensor data 引用。

关于 TensorData 类型的定义，请参见 [15.2.2.32 TensorData](#152232-tensordata)。

## 约束说明
无。

## 调用示例
```cpp
StorageShape sh({1, 2, 3}, {1, 2, 3});
Tensor t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
auto a = reinterpret_cast<void *>(10);
t.MutableTensorData() = TensorData{a, nullptr}; // 设置新 tensordata
auto td = t.GetTensorData(); // TensorData{a, nullptr}
```
