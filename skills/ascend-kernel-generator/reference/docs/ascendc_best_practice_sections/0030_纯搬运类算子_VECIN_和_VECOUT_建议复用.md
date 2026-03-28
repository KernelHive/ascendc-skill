### 纯搬运类算子 VECIN 和 VECOUT 建议复用

【优先级】高

【描述】
纯搬运类算子在执行时并不涉及实际 vector 计算，若存在冗余的 vector 指令，会导致算子整体执行时间变长。这种场景可以使用 Ascend C 针对纯搬运类算子提供的 TQueBind 接口，该接口可以将 VECIN 与 VECOUT 绑定，省略将数据从 VECIN 拷贝到 VECOUT 的步骤，从而避免 vector 的无谓消耗。

【反例】
此段代码为了保证搬入和搬出之间的流水同步，存在 LocalTensor → LocalTensor 的 DataCopy 指令。

```cpp
template <typename ComputeT> class KernelExample {
public:
    ...
    __aicore__ inline void Process(...)
    {
        for (int i = 0; i < iLen; ++i) {
            ...
            auto iLocal = QueI.AllocTensor<ComputeT>();
            DataCopy(iLocal, inGm[i * 32], size);
            QueI.EnQue(iLocal);
            iLocal = QueI.DeQue<ComputeT>();
            for (int j = 0; j < jLen; ++j) {
                ...
                auto oLocal = QueO.AllocTensor<ComputeT>();
                DataCopy(oLocal, iLocal, size); // LocalTensor -> LocalTensor的DataCopy指令,以实现数据从VECIN到VECOUT的搬运
                QueO.EnQue(oLocal);

                auto oLocal = QueO.DeQue<ComputeT>();
                DataCopyPad(outGm[j], oLocal, ...);
                QueO.FreeTensor(oLocal);
            }
            QueI.FreeTensor(iLocal);
        }
    }

private:
    ...
    TQue<TPosition::VECIN, BUFFER_NUM> QueI;
    TQue<TPosition::VECOUT, BUFFER_NUM> QueO;
    ...
};

extern "C" __global__ __aicore__ void example_kernel(...)
{
    ...
    op.Process(...);
}
```

【正例】
将 LocalTensor → LocalTensor 的 DataCopy 指令替换为 TQueBind 接口，避免将 VECIN 拷贝到 VECOUT 的步骤，从而避免了冗余拷贝。

```cpp
template <typename ComputeT> class KernelExample {
public:
    ...
    __aicore__ inline void Process(...)
    {
        for (int i = 0; i < iLen; ++i) {
            ...
            auto bindLocal = queBind.AllocTensor<ComputeT>();
            DataCopy(bindLocal, inGm[i * 32], size);
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<ComputeT>();
            for (int j = 0; j < jlen; ++j) {
                ...
                DataCopyPad(outGm[j], bindLocal, ...);
            }
            queBind.FreeTensor(bindLocal);
        }
    }

private:
    ...
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queBind; // 使用TQueBind替换原来的QueI，QueO
    ...
};

extern "C" __global__ __aicore__ void example_kernel(...)
{
    ...
    op.Process(...);
}
```

【性能对比】

![图 5-17 aiv_vec_time 优化前后对比]()

如上图所示，将反例中 DataCopy 指令替换为 TQueBind 之后有明显优化。由于省略了数据从 VECIN 拷贝到 VECOUT 的步骤，aiv_vec_time 几乎缩减为 0。
