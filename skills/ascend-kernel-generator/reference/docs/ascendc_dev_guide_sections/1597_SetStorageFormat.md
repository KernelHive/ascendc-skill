##### SetStorageFormat

## 函数功能
设置运行时 Tensor 的存储格式。

## 函数原型
```cpp
void SetStorageFormat(const ge::Format storage_format)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| storage_format | 输入 | 运行时格式。<br>关于 ge::Format 类型的定义，请参见 15.2.3.59 Format。 |

## 返回值说明
无。

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
