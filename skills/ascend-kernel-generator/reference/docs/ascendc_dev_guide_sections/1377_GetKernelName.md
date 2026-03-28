##### GetKernelName

## 函数功能
获取本 kernel 的名称。

## 函数原型
```cpp
const ge::char_t *GetKernelName() const
```

## 参数说明
无。

## 返回值说明
本 kernel 的 name。

## 约束说明
无。

## 调用示例
```cpp
// 假设已存在 KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
auto kernel_name = extend_context->GetKernelName();
```
