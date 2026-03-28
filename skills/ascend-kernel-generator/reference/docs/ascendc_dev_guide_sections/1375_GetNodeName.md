##### GetNodeName

## 函数功能
获取算子的名称。

## 函数原型
```cpp
const ge::char_t *GetNodeName() const
```

## 参数说明
无。

## 返回值说明
算子的名称。

## 约束说明
无。

## 调用示例
```cpp
// 假设已存在KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
auto node_name = extend_context->GetNodeName();
```
