##### GetStorageFormat

## 函数功能
获取运行时 Tensor 的 format。

## 函数原型
```cpp
ge::Format GetStorageFormat() const
```

## 参数说明
无。

## 返回值说明
返回运行时 format。

关于 `ge::Format` 类型的定义，请参见 15.2.3.59 Format。

## 约束说明
无。

## 调用示例
```cpp
StorageShape sh({1, 2, 3}, {1, 2, 3});
Tensor t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
t.SetOriginFormat(ge::FORMAT_NHWC);

t.SetStorageFormat(ge::FORMAT_NC1HWC0);
auto fmt = t.GetStorageFormat(); // ge::FORMAT_NC1HWC0
```
