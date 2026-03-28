##### GetExtendInfo

## 函数功能
获取本 kernel 的扩展信息。

## 函数原型
```cpp
const KernelExtendInfo *GetExtendInfo() const
```

## 参数说明
无。

## 返回值说明
本 kernel 的扩展信息。

## 约束说明
无。

## 调用示例
```cpp
// 假设已存在 KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
auto extend_info = extend_context->GetExtendInfo();
```

> 关于 `KernelExtendInfo` 类型的定义，请参见 15.2.2.39 内部关联接口 `KernelExtendInfo` 类。
