##### GetComputeNodeOutputNum

## 函数功能
获取算子的输出个数。

## 函数原型
```cpp
size_t GetComputeNodeOutputNum() const
```

## 参数说明
无。

## 返回值说明
算子的输出个数。

## 约束说明
无。

## 调用示例
```cpp
// 假设已存在KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
for (size_t idx = 0; idx < extend_context->GetComputeNodeOutputNum(); ++idx) {
    auto input_td = extend_context->GetOutputDesc(idx);
    // ...
}
```
