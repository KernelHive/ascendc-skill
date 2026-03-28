##### GetOutputShape

## 函数功能

根据算子输出索引获取对应的输出 shape 指针。这里的输出索引是指算子实例化后的实际索引，不是原型定义中的索引。

## 函数原型

```cpp
const StorageShape *GetOutputShape(size_t index) const
```

## 参数说明

| 参数   | 输入/输出 | 说明                               |
|--------|-----------|------------------------------------|
| index  | 输入      | 算子输出索引，从 0 开始计数。      |

## 返回值说明

指定的输出 shape 指针，index 非法时，返回空指针。

关于 StorageShape 类型的定义，请参见 15.2.2.29 StorageShape。

## 约束说明

无。

## 调用示例

```cpp
ge::graphStatus Tiling4ConcatD(TilingContext* context) {
  auto out_shape = context->GetOutputShape(0);
  // ...
}
```
