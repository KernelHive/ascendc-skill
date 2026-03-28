##### GetDynamicInputShape

## 函数功能

根据算子原型定义中的输入索引获取对应的动态输入 shape 指针。

## 函数原型

```cpp
const StorageShape *GetDynamicInputShape(const size_t ir_index, const size_t relative_index) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| ir_index | 输入 | 动态输入在算子 IR 原型定义中的索引，从 0 开始计数 |
| relative_index | 输入 | 该输入实例化后的相对 index，例如某个 `DYNAMIC_INPUT` 实例化了 3 个输入，那么 relative_index 的取值范围是 [0,2] |

## 返回值说明

指定的输入 shape 指针，shape 中包含了原始 shape 与运行时 shape。关于 StorageShape 类型的定义，请参见 15.2.2.29 StorageShape。

当输入 ir_index 或 relative_index 非法时，返回空指针。

## 约束说明

无。

## 调用示例

```cpp
ge::graphStatus InferShape4ConcatD(TilingContext* context) {
  auto in_shape = context->GetDynamicInputShape(dynamic_input_idx, 0);
  // ...
}
```
