##### GetDynamicOutputFormat

## 函数功能

根据算子原型定义中的输出索引获取对应的动态输出 Format 指针。

## 函数原型

```cpp
StorageFormat *GetDynamicOutputFormat(const size_t ir_index, const size_t relative_index)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| ir_index | 输入 | IR 原型定义中的索引，从 0 开始计数。 |
| relative_index | 输入 | 该算子输出实例化后的相对 index，例如某个动态输出实例化了 3 个输出，那么 relative_index 的有效范围是 [0,2]。 |

## 返回值说明

返回输出 Format 指针，ir_index 或 relative_index 非法时，返回空指针。

关于 StorageFormat 类型的定义，请参见 15.2.2.28 StorageFormat。

## 约束说明

无。

## 调用示例

```cpp
ge::graphStatus InferFormatForXXX(InferFormatContext *context) {
  auto format = context->GetDynamicOutputFormat(0U, 0U);
  // ...
}
```
