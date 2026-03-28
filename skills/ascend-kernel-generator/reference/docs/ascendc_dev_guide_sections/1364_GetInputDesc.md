##### GetInputDesc

## 函数功能

根据算子输入索引获取对应输入的 tensor 描述信息。这里的输入索引是指算子实例化后实际的索引，不是原型定义中的索引。

## 函数原型

```cpp
const CompileTimeTensorDesc *GetInputDesc(const size_t index) const
```

## 参数说明

| 参数   | 输入/输出 | 说明                           |
|--------|-----------|--------------------------------|
| index  | 输入      | 算子输入索引，从 0 开始计数。 |

## 返回值说明

输入 TensorDesc 的指针，当输入 index 非法时，返回空指针。

关于 `CompileTimeTensorDesc` 的定义，请参见 15.2.2.2 CompileTimeTensorDesc。

## 约束说明

无。

## 调用示例

```cpp
// 假设已存在 KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
for (size_t idx = 0; idx < extend_context->GetComputeNodeInputNum(); ++idx) {
    auto input_td = extend_context->GetInputDesc(idx);
    // ...
}
```
