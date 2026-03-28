##### GetInputShape

## 函数功能

根据算子输入索引获取对应的输入 Shape 指针。这里的输入索引是指算子实例化后实际的索引，不是原型定义中的索引。

## 函数原型

```cpp
const Shape *GetInputShape(const size_t index) const
```

## 参数说明

| 参数   | 输入/输出 | 说明                           |
| ------ | --------- | ------------------------------ |
| `index` | 输入      | 算子输入索引，从 0 开始计数。 |

## 返回值说明

输入 Shape 指针，`index` 非法时，返回空指针。

> 关于 Shape 类型的定义，请参见 15.2.2.27 Shape。

## 约束说明

无。

## 调用示例

```cpp
ge::graphStatus InferFormatForXXX(InferFormatContext *context) {
  const auto shape = context->GetInputShape(0); // 获取第 0 个输入的 shape
  GE_ASSERT_NOTNULL(shape);
  // ...
}
```
