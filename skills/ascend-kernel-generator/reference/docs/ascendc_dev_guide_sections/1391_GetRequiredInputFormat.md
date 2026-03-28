##### GetRequiredInputFormat

## 函数功能

根据算子原型定义中的输入索引获取对应的必选输入 Format 指针。

## 函数原型

```cpp
StorageFormat *GetRequiredInputFormat(const size_t ir_index)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| ir_index | 输入 | IR 原型定义中的索引，从 0 开始计数 |

## 返回值说明

输入 Format 指针，ir_index 非法时，返回空指针。

> 关于 StorageFormat 类型的定义，请参见 15.2.2.28 StorageFormat。

## 约束说明

无。

## 调用示例

```cpp
ge::graphStatus InferFormatForXXX(InferFormatContext *context) {
    auto format = context->GetRequiredInputFormat(0); // 获取第 0 个输入的 format
    GE_ASSERT_NOTNULL(format);
    // ...
}
```
