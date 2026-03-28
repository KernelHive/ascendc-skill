##### GetOriginFormat

## 功能
获取 Tensor 的原始格式。

## 原型
```cpp
ge::Format GetOriginFormat() const
```

## 参数
无。

## 返回值
原始格式。

> 关于 `ge::Format` 类型的定义，请参见 [15.2.3.59 Format](#)。

## 约束
无。

## 示例
```cpp
StorageShape sh({1, 2, 3}, {1, 2, 3});
Tensor t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
t.SetOriginFormat(ge::FORMAT_NHWC);
t.SetStorageFormat(ge::FORMAT_NC1HWC0);
auto fmt = t.GetOriginFormat(); // ge::FORMAT_NHWC
```
