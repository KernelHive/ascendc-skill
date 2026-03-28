##### GetComputeNodeInfo

## 函数功能

获取本 Kernel 对应的计算节点的信息。

图执行时本质上是执行图上的一个个节点的 Kernel 在执行。本方法能够从 KernelContext 中获取保存的 ComputeNodeInfo，而 ComputeNodeInfo 中包含 InputDesc 等信息。

## 函数原型

```cpp
const ComputeNodeInfo *GetComputeNodeInfo() const
```

## 参数说明

无。

## 返回值说明

计算节点的信息。

关于 ComputeNodeInfo 的定义，请参见 15.2.2.3 ComputeNodeInfo。

## 约束说明

无。

## 调用示例

```cpp
// 假设已存在 KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
auto compute_node_info = extend_context->GetComputeNodeInfo();
```
