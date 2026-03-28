##### GetDataType

## 函数功能
获取 Tensor 的数据类型。

## 函数原型
```cpp
ge::DataType GetDataType() const
```

## 参数说明
无。

## 返回值说明
返回 Tensor 中的数据类型。

关于 `ge::DataType` 的定义，请参见 [15.2.3.58 DataType](#152358-datatype)。

## 约束说明
无。

## 调用示例
```cpp
StorageShape sh({1, 2, 3}, {1, 2, 3});
Tensor t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
auto dt = t.GetDataType(); // ge::DT_FLOAT
```
