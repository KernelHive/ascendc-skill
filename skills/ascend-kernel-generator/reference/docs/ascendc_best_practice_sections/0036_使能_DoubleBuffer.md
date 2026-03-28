### 使能 DoubleBuffer

【优先级】中

## 描述

执行于 AI Core 上的指令队列主要包括以下几类：

- Vector 指令队列（V）
- Cube 指令队列（M）
- Scalar 指令队列（S）
- 搬运指令队列（MTE1/MTE2/MTE3）

不同指令队列间的相互独立性和可并行执行特性，是 DoubleBuffer 优化机制的基石。

以纯 Vector 计算为例：

- 矢量计算前后的 CopyIn、CopyOut 过程使用搬运指令队列（MTE2/MTE3）
- Compute 过程使用 Vector 指令队列（V）

不同指令队列可并行执行，意味着 CopyIn、CopyOut 过程和 Compute 过程是可以并行的。

如图 5-24 所示，考虑一个完整的数据搬运和计算过程：

- CopyIn 过程将数据从 Global Memory 搬运到 Local Memory
- Vector 计算单元完成 Compute 计算后，经过 CopyOut 过程将计算结果搬回 Global Memory

图 5-24 数据搬运与 Vector 计算过程

图 5-25 未使能 DoubleBuffer 的流水图

在此过程中，数据搬运与 Vector 计算串行执行，Vector 计算单元不可避免存在资源闲置问题。假设 CopyIn、Compute、CopyOut 三阶段分别耗时相同均为 t，则 Vector 的利用率仅为 1/3，等待时间过长，Vector 利用率严重不足。

为减少 Vector 等待时间，使能 DoubleBuffer 机制将待处理的数据一分为二，比如 Tensor1、Tensor2。

如图 5-26 所示：

- 当 Vector 单元对 Tensor1 中数据进行 Compute 计算时，Tensor2 数据流可以执行 CopyIn 的过程
- 当 Vector 切换到计算 Tensor2 时，Tensor1 数据流可以执行 CopyOut 的过程

由此，数据的进出搬运和 Vector 计算实现并行执行，Vector 闲置问题得以有效缓解。

图 5-26 DoubleBuffer 机制

图 5-27 使能 DoubleBuffer 的流水图

总体来说，DoubleBuffer 是基于 MTE 指令队列与 Vector 指令队列的独立性和可并行性，通过将数据搬运与 Vector 计算并行执行以隐藏大部分的数据搬运时间，并降低 Vector 指令的等待时间，最终提高 Vector 单元的利用效率。

通过为队列申请内存时设置内存块的个数为 2，使能 DoubleBuffer，实现数据并行，简单代码示例如下：

```cpp
pipe.InitBuffer(inQueueX, 2, 256);
```

## 需要注意

多数情况下，采用 DoubleBuffer 能有效提升 Vector 的利用率，缩减算子执行时间。然而，DoubleBuffer 机制缓解 Vector 闲置问题，并不代表它总能带来明显的整体性能提升。例如：

- 当数据搬运时间较短，而 Vector 计算时间显著较长时，由于数据搬运在整个计算过程中的时间占比较低，DoubleBuffer 机制带来的性能收益会偏小
- 当原始数据较小且 Vector 可一次性完成所有数据量的计算时，强行使用 DoubleBuffer 会降低 Vector 计算资源的利用率，最终效果可能适得其反

因此，DoubleBuffer 的使用需综合考虑 Vector 算力、数据量大小、搬运与计算时间占比等多种因素。

## 反例

```cpp
__aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
{
    src0Global.SetGlobalBuffer((__gm__ half*)src0Gm);
    src1Global.SetGlobalBuffer((__gm__ half*)src1Gm);
    dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
    // 不使能 DoubleBuffer，占用的物理空间是 1 * sizeSrc0 * sizeof(half)
    // 3 个 InitBuffer 执行后总空间为 1 * (sizeSrc0 * sizeof(half) + sizeSrc1 * sizeof(half) + sizeDst0 * sizeof(half))
    pipe.InitBuffer(inQueueSrc0, 1, sizeSrc0 * sizeof(half));
    pipe.InitBuffer(inQueueSrc1, 1, sizeSrc1 * sizeof(half));
    pipe.InitBuffer(outQueueDst, 1, sizeDst0 * sizeof(half));
}

__aicore__ inline void Process()
{
    // 需要 round * 2 次循环才能处理完数据
    for (uint32_t index = 0; index < round * 2; ++index) {
        CopyIn(index);
        Compute();
        CopyOut(index);
    }
}
```

## 正例

```cpp
__aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
{
    src0Global.SetGlobalBuffer((__gm__ half*)src0Gm);
    src1Global.SetGlobalBuffer((__gm__ half*)src1Gm);
    dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
    // InitBuffer 中使用 2 表示使能 DoubleBuffer，占用的物理空间是 2 * sizeSrc0 * sizeof(half)
    // 3 个 InitBuffer 执行后总空间为 2 * (sizeSrc0 * sizeof(half) + sizeSrc1 * sizeof(half) + sizeDst0 * sizeof(half))
    pipe.InitBuffer(inQueueSrc0, 2, sizeSrc0 * sizeof(half));
    pipe.InitBuffer(inQueueSrc1, 2, sizeSrc1 * sizeof(half));
    pipe.InitBuffer(outQueueDst, 2, sizeDst0 * sizeof(half));
}

__aicore__ inline void Process()
{
    // 开启 DoubleBuffer 的前提是循环次数 >= 2
    for (uint32_t index = 0; index < round; ++index) {
        CopyIn(index);
        Compute();
        CopyOut(index);
    }
}
```
