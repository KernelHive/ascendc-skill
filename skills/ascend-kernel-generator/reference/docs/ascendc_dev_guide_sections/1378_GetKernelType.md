##### GetKernelType

## 函数功能
获取本 kernel 的类型。

## 函数原型
```cpp
const ge::char_t *GetKernelType() const
```

## 参数说明
无。

## 返回值说明
本 kernel 的 type。

## 约束说明
无。

## 调用示例
```cpp
// 假设已存在 KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
auto kernel_type = extend_context->GetKernelType();
```
