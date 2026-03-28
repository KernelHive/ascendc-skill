###### KernelRunContextHolder 结构定义

## 功能说明

该结构体为 ContextBuilder 类最终的构造结果，可通过指定的接口获取内部算子信息或获取 KernelContext 类的对象。

## 定义原型

```cpp
struct KernelRunContextHolder {
    KernelRunContextHolder();
    ~KernelRunContextHolder();
    
    template<typename T>
    T *GetContext() const
    {
        return reinterpret_cast<T*>(context);
    }
    
    gert::ComputeNodeInfo *MutableComputeNodeInfo()
    {
        return reinterpret_cast<gert::ComputeNodeInfo *>(computeNodeExtendHolder.get());
    }
    
    std::unique_ptr<ValueHolderImpl> valueHolder;
    std::unique_ptr<uint8_t[]> computeNodeExtendHolder;
    KernelRunContext *context {nullptr};
};
```

## 函数说明

| 函数名称 | 入参说明 | 含义 |
|---------|---------|------|
| GetContext | 无 | 获取 context 成员变量转化为模板 T 的指针，T 可选值为 KernelContext 以及它的子类如 TilingContext |
| MutableComputeNodeInfo | 无 | 返回构造的 gert::ComputeNodeInfo 类指针 |

## 变量说明

| 变量名称 | 变量含义 |
|---------|---------|
| valueHolder | 保证 KernelRunContextHolder 内部值不析构的智能指针 |
| computeNodeExtendHolder | 可转化成 ComputeNodeInfo 类的智能指针 |
| context | 指向 KernelRunContext 类的指针 |

## 约束说明

无

## 调用示例

```cpp
auto holder = context_ascendc::ContextBuilder().Inputs().Outputs().BuildKernelRunContext();
if (holder != nullptr) {
    gert::KernelContext* tilingParseContext = holder->GetContext<gert::KernelContext>();
    gert::ComputeNodeInfo* info = holder->MutableComputeNodeInfo();
}
```
