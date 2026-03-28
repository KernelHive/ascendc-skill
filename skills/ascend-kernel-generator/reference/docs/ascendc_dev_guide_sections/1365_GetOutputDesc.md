##### GetOutputDesc

## 函数功能

根据算子输出索引获取对应输出的 tensor 描述信息。这里的输出索引是指算子实例化后实际的索引，不是原型定义中的索引。

## 函数原型

```cpp
const CompileTimeTensorDesc *GetOutputDesc(const size_t index) const
```

## 参数说明

| 参数   | 输入/输出 | 说明                               |
| ------ | --------- | ---------------------------------- |
| `index` | 输入      | 算子输出索引，从 0 开始计数。 |

## 返回值说明

输出 `TensorDesc` 的指针，当输入 `index` 非法时，返回空指针。

> 关于 `CompileTimeTensorDesc` 的定义，请参见 15.2.2.2 CompileTimeTensorDesc。

## 约束说明

无。

## 调用示例

```cpp
// 假设已存在 KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
for (size_t idx = 0; idx < extend_context->GetComputeNodeInfo()->GetOutputsNum(); ++idx) {
    auto output_td = extend_context->GetOutputDesc(idx);
    // ...
}
```
