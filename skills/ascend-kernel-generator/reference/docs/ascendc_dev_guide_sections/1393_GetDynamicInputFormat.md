##### GetDynamicInputFormat

## 函数功能

根据算子原型定义中的输入索引获取对应的动态输入 Format 指针。

## 函数原型

```cpp
StorageFormat *GetDynamicInputFormat(const size_t ir_index, const size_t relative_index)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| ir_index | 输入 | IR 原型定义中的 index |
| relative_index | 输入 | 该输入实例化后的相对 index，例如某个动态输入实例化了 3 个输入，那么 relative_index 的有效范围是 [0,2] |

## 返回值说明

输入 Format 指针，ir_index 或 relative_index 非法时，返回空指针。

关于 StorageFormat 类型的定义，请参见 15.2.2.28 StorageFormat。

## 约束说明

无。

## 调用示例

```cpp
ge::graphStatus InferFormatForXXX(InferFormatContext *context) {
    const auto format = context->GetDynamicInputFormat(1U, 0U);
    GE_ASSERT_NOTNULL(format);
    // ...
}
```
