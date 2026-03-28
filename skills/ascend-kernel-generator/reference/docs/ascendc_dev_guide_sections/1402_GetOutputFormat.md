##### GetOutputFormat

## 函数功能

根据算子输出索引获取对应的输出 Format 指针。这里的输出索引是指算子实例化后实际的索引，不是原型定义中的索引。

## 函数原型

```cpp
StorageFormat *GetOutputFormat(const size_t index)
```

## 参数说明

| 参数   | 输入/输出 | 说明                                   |
|--------|-----------|----------------------------------------|
| index  | 输入      | 算子输出索引，从 0 开始计数。          |

## 返回值说明

输出 Format 指针，index 非法时，返回空指针。

关于 StorageFormat 类型的定义，请参见 15.2.2.28 StorageFormat。

## 约束说明

无。

## 调用示例

```cpp
ge::graphStatus InferFormatForXXX(InferFormatContext *context) {
  auto format = context->GetOutputFormat(0U); // 获取第 0 个输出的 format
  // ...
}
```
