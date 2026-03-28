###### BuildKernelRunContext

## 功能说明

构造 KernelRunContext 并返回 KernelRunContextHolder 的智能指针，该对象可通过 GetContext 接口获取 KernelContext 类型的对象。

## 函数原型

```cpp
std::shared_ptr<KernelRunContextHolder> BuildKernelRunContext()
```

## 参数说明

无

## 返回值说明

KernelRunContextHolder 的共享指针，可通过 `GetContext<gert::KernelContext>()` 函数获取 KernelContext 对象。

## 约束说明

无

## 调用示例

```cpp
auto kernelContextHolder = context_ascendc::ContextBuilder().Inputs().Outputs().BuildKernelRunContext();
gert::KernelContext* tilingParseContext = kernelContextHolder->GetContext<gert::KernelContext>();
```
