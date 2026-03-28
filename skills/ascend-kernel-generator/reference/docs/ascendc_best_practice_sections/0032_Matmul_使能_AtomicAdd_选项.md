### Matmul 使能 AtomicAdd 选项

## 优先级
中

## 描述
对于 Matmul 得到的结果矩阵 C(m, n)，若后续需要和 GM 上的矩阵 D(m, n) 进行 Add 操作，则可以在 `GetTensorC` 接口或者 `IterateAll` 接口的 GM 通路上，将 `enAtomic` 参数设为 1，开启 AtomicAdd 累加操作。在搬出矩阵 C 到 GM 时，矩阵 C 的结果将直接累加到矩阵 D 的 GM 地址上，从而实现与矩阵 D 的 Add 操作。

## 反例
将 Matmul 的结果矩阵 C 和 GM 上的矩阵 D 分别搬到 UB 上，做完 Add 操作后，结果再搬出到 GM。这样至少要多分配一块 UB 内存给矩阵 D，假设在分离架构的处理器上执行，将多做三次搬运操作（矩阵 C 从 GM 搬到 UB、矩阵 D 从 GM 搬到 UB、Add 结果从 UB 搬出到 GM）。

```cpp
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatMulKernel(...)
{
    ...
    AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> mm;
    TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);

    mm.SetTensorA(gm_a);
    mm.SetTensorB(gm_b);
    mm.SetBias(gm_bias);
    mm.IterateAll(gm_c);

    DataCopy(local_c, gm_c, c_size);
    DataCopy(local_d, gm_d, d_size);
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    Add(local_d, local_d, local_c, d_size);
    DataCopy(gm_d, local_d, d_size);
    ...
}

extern "C" __global__ __aicore__ void example_kernel(...)
{
    ...
    typedef AscendC::MatmulType<TPosition::GM, CubeFormat::ND, half> aType;
    typedef AscendC::MatmulType<TPosition::GM, CubeFormat::ND, half> bType;
    typedef AscendC::MatmulType<TPosition::GM, CubeFormat::ND, float> cType;
    typedef AscendC::MatmulType<TPosition::GM, CubeFormat::ND, float> biasType;
    MatMulKernel<aType, bType, cType, biasType>(...);
    ...
}
```

## 正例
计算 Matmul 结果时，调用 `IterateAll` 接口或者 `GetTensorC` 接口搬运到矩阵 D 的 GM 地址上，同时将接口中 `enAtomic` 参数设为 1。搬出到 GM 时，Matmul 结果矩阵 C 会累加到矩阵 D 上，从而得到两个矩阵 Add 后的结果。

```cpp
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatMulKernel(...)
{
    ...
    AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> mm;
    TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);

    mm.SetTensorA(gm_a);
    mm.SetTensorB(gm_b);
    mm.SetBias(gm_bias);

    mm.IterateAll(gm_d, 1); // IterateAll接口中的enAtomic设为1
    // while (mm.Iterate()) {
    //     mm.GetTensorC(gm_d, 1); // GetTensorC接口中的enAtomic设为1
    // }
    ...
}

extern "C" __global__ __aicore__ void example_kernel(...)
{
    ...
    typedef AscendC::MatmulType<TPosition::GM, CubeFormat::ND, half> aType;
    typedef AscendC::MatmulType<TPosition::GM, CubeFormat::ND, half> bType;
    typedef AscendC::MatmulType<TPosition::GM, CubeFormat::ND, float> cType;
    typedef AscendC::MatmulType<TPosition::GM, CubeFormat::ND, float> biasType;
    MatMulKernel<aType, bType, cType, biasType>(...);
    ...
}
```

## 性能对比
以矩阵维度 M=64，N=256，K=256，矩阵 D 为 (64, 256) 为例，Matmul 使能 AtomicAdd 选项前后的性能对比如下图所示，平均 cycle 数从开启 AtomicAdd 选项前的 154181 变为开启后的 135054，性能优化 12.4%。因此在这种场景下，使能 AtomicAdd 选项能获取更优的性能。

![图 5-20 Matmul 使能 AtomicAdd 选项前后性能对比](image.png)
