##### SetOriginFormat

## 功能
设置 Tensor 的原始格式。

## 原型
```cpp
void SetOriginFormat(const ge::Format origin_format)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| origin_format | 输入 | 原始格式。 |

> 关于 `ge::Format` 类型的定义，请参见 15.2.3.59 Format。

## 返回值
无。

## 约束
无。

## 调用示例
```cpp
StorageShape sh({1, 2, 3}, {1, 2, 3});
Tensor t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
t.SetOriginFormat(ge::FORMAT_NHWC);
t.SetStorageFormat(ge::FORMAT_NC1HWC0);
auto fmt = t.GetOriginFormat(); // ge::FORMAT_NHWC
```
