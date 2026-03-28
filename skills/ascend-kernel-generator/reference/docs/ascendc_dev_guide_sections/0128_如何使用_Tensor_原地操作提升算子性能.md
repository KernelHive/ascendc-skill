## 如何使用 Tensor 原地操作提升算子性能

Tensor 原地操作（inplace 接口）是一种优化技术，全局申请、保留 LocalTensor 内存，避免了频繁创建和销毁 LocalTensor 对象。AllocTensor、FreeTensor、EnQue、DeQue 接口不产生新的 LocalTensor，而是在该全局 LocalTensor 上反复申请、释放、入队、出队。其实现原理如下图所示：

图 12-8 Tensor 原地操作实现原理

## Tensor 原地操作的优势

- **减少栈变换**：相比构造新 Tensor 的方式，inplace 接口减少了 LocalTensor 的栈变换，允许 Tensor 被反复使用。
- **减少入队/出队操作**：在调用 EnQue、DeQue 的过程中，TQue 对象没有存储该 Tensor 对应的 Buffer 地址，实际没有真正入队、出队，减少了反复入队、出队的 Scalar 指令。

## 保留 EnQue 和 DeQue 的原因

既然 Tensor 原地操作没有执行真正的入队出队操作，为什么还需要保留 EnQue 和 DeQue 接口呢？

- **编程兼容性**：为了保持编程接口的一致性，inplace 接口仍然需要调用 EnQue 和 DeQue，确保代码结构的统一性和可维护性。
- **内存同步功能**：EnQue 和 DeQue 操作中实现了内存读写同步功能，确保数据的一致性和正确性，即使没有实际的队列操作，这些同步机制仍然需要保留。

## 适用场景

适合计算循环次数多的场景：如图 12-8 所示，inplace 接口虽然增加了 TQue 对象 InitBuffer 的初始化开销，但显著减少了每次循环中 AllocTensor、EnQue、DeQue 和 FreeTensor 内部对 LocalTensor 和事件的操作次数，特别适合需要多次循环来完成计算的场景。

## 使用方法

- **配置 TQue 对象**：在创建 TQue 对象时，设置深度（depth）为 0，启用 inplace 操作模式。
- **调用原地操作接口**：使用 inplace 接口直接操作 LocalTensor。
  - AllocTensor 和 DeQue 区分 non-inplace 和 inplace 接口，详情请参考 AllocTensor、DeQue。
  - FreeTensor 和 EnQue 不区分 non-inplace 和 inplace 接口。

## 示例代码

```cpp
// ...
namespace AscendC {
class MyKernel {
public:
    __aicore__ inline MyKernel() {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
    {
        src0Global.SetGlobalBuffer((__gm__ half*)src0Gm);
        src1Global.SetGlobalBuffer((__gm__ half*)src1Gm);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        pipe.InitBuffer(srcQue0, 1, BLOCK_SIZE * sizeof(half));
        pipe.InitBuffer(srcQue1, 1, BLOCK_SIZE * sizeof(half));
        pipe.InitBuffer(dstQue0, 1, BLOCK_SIZE * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (int i = 0; i < REPTIMES; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t i)
    {
        srcQue0.AllocTensor<half>(src0Local);
        srcQue1.AllocTensor<half>(src1Local);
        DataCopy(src0Local, src0Global[i*BLOCK_SIZE], BLOCK_SIZE);
        DataCopy(src1Local, src1Global[i*BLOCK_SIZE], BLOCK_SIZE);
        srcQue0.EnQue(src0Local);
        srcQue1.EnQue(src1Local);
    }
    __aicore__ inline void Compute(int32_t i)
    {
        srcQue0.DeQue<half>(src0Local);
        srcQue1.DeQue<half>(src1Local);
        dstQue0.AllocTensor<half>(dstLocal);
        Add(dstLocal, src0Local, src1Local, BLOCK_SIZE);
        dstQue0.EnQue<half>(dstLocal);
        srcQue0.FreeTensor(src0Local);
        srcQue1.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut(int32_t i)
    {
        dstQue0.DeQue<half>(dstLocal);
        DataCopy(dstGlobal[i*BLOCK_SIZE], dstLocal, BLOCK_SIZE);
        dstQue0.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 0> srcQue0, srcQue1;
    TQue<QuePosition::VECOUT, 0> dstQue0;
    GlobalTensor<half> src0Global, src1Global, dstGlobal;
    LocalTensor<half> src0Local;
    LocalTensor<half> src1Local;
    LocalTensor<half> dstLocal;
};
} // namespace AscendC
// ...
```
