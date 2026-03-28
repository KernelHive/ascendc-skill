##### GetNodeType

## 函数功能
获取算子的类型。

## 函数原型
```cpp
const ge::char_t *GetNodeType() const
```

## 参数说明
无。

## 返回值说明
算子的类型。

## 约束说明
无。

## 调用示例
```cpp
// 假设已存在KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
auto node_type = extend_context->GetNodeType();
```
