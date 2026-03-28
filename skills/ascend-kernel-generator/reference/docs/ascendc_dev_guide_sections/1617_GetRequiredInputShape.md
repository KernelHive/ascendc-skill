##### GetRequiredInputShape

## 函数功能

根据算子原型定义中的输入索引获取对应的必选输入 shape 指针。

## 函数原型

```cpp
const StorageShape *GetRequiredInputShape(const size_t ir_index) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| ir_index | 输入 | 必选输入在算子 IR 原型定义中的索引，从 0 开始计数 |

## 返回值说明

指定的输入 shape 指针，shape 中包含了原始 shape 与运行时 shape。关于 StorageShape 类型的定义，请参见 15.2.2.29 StorageShape。

当输入 ir_index 非法时，返回空指针。

## 约束说明

无。

## 调用示例

```cpp
ge::graphStatus InferShape4ConcatD(TilingContext* context) {
  auto in_shape = context->GetRequiredInputShape(0);
  // ...
}
```
