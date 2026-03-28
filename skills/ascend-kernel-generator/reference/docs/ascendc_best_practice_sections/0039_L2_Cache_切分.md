### L2 Cache 切分

## 优先级
高

## 描述
假设 AI 处理器的 L2 Cache 大小为 192MB，L2 Cache 读写混合带宽约为 7TB/s，而 GM 的带宽约为 1.6TB/s，两者之间存在较大差距。搬入或搬出相同数据量的情况下，访问 L2 Cache 读写数据比 GM 更快。

若数据无法命中 L2 Cache，即需要访问的数据不在 L2 Cache 内，导致需要去 GM 上读写，带宽利用效率较低，最终算子搬入或搬出数据变为算子整个运行过程的性能瓶颈。

**切分策略建议**：当输入和输出数据的数据量超过 L2 Cache 大小时，Tiling 中使能 L2 Cache 切分策略。

## 反例
假设输入数据大小为 `InputTotalSize = 384MB`，L2 Cache 大小为 192MB，总核数为 20 个核，数据未切分，整体一次完成计算。假设 20 个核一次可以处理共 192MB 的数据，则每个核至少两次读取输入数据。

![图 5-30 未使能 L2 Cache 切分]()

```cpp
constexpr int32_t TOTAL_LENGTH = 384 * 1024 * 1024 / sizeof(half);
constexpr int32_t USE_CORE_NUM = 20;
constexpr int32_t TILE_NUM = 2;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM;

class KernelSample {
public:
    __aicore__ inline KernelSample() {}
    __aicore__ inline void Init(GM_ADDR x)
    {
        xGm.SetGlobalBuffer((__gm__ half*)x + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH);
        pipe.InitBuffer(inQueueX, 1, BLOCK_LENGTH * sizeof(half));
        pipe.InitBuffer(inQueueY, 1, BLOCK_LENGTH * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        // 示例演示对输入数据加2的运算
        constexpr int32_t loopCount = 2;
        for (int32_t i = 0; i < loopCount; i++) {
            // 外层的每次循环对输入数据进行加1的运算
            for (int32_t j = 0; j < TILE_NUM; j++) {
                // 内层循环分别处理每个核第0块和第1块数据
                CopyIn(j);
                Compute();
                CopyOut(j);
            }
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t process)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        // 对于每个核，除了首次读取外，读取第0块数据时，L2 Cache内缓存的是第1块数据；
        // 对于每个核，读取第1块数据时，L2 Cache内缓存的是第0块数据；
        // 每个核需要4次读取GM上的数据
        DataCopy(xLocal, xGm[process * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        Adds(yLocal, xLocal, 1, TILE_LENGTH);
        inQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t process)
    {
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        DataCopy(yGm[process * TILE_LENGTH], yLocal, TILE_LENGTH);
        inQueueY.FreeTensor(yLocal);
    }
};
```

## 正例
假设输入数据大小为 `InputTotalSize = 384MB`，L2 Cache 大小为 192MB，能使用的总核数为 20 个核，输入数据均等切分成 2 份数据，则整体分两次进行计算，每次的计算量为 192MB。第一次 20 个核先计算前 192MB 的数据，第二次 20 个核计算后 192MB 的数据。每次计算前读取的数据能够命中 L2 Cache，提升算子性能。

![图 5-31 使能 L2 Cache 切分]()

```cpp
constexpr int32_t TOTAL_LENGTH = 384 * 1024 * 1024 / sizeof(half);
constexpr int32_t TILE_NUM = 2;
constexpr int32_t USE_CORE_NUM = 20;
constexpr int32_t TILE_LENGTH = TOTAL_LENGTH / TILE_NUM;
constexpr int32_t BLOCK_LENGTH = TILE_LENGTH / USE_CORE_NUM;

class KernelSample {
public:
    __aicore__ inline KernelSample() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int32_t index)
    {
        xGm.SetGlobalBuffer((__gm__ half*)x + BLOCK_LENGTH * GetBlockIdx() + index * TILE_LENGTH,
                            BLOCK_LENGTH);
        yGm.SetGlobalBuffer((__gm__ half*)y + BLOCK_LENGTH * GetBlockIdx() + index * TILE_LENGTH,
                            BLOCK_LENGTH);
        pipe.InitBuffer(inQueueX, 1, BLOCK_LENGTH * sizeof(half));
        pipe.InitBuffer(inQueueY, 1, BLOCK_LENGTH * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        // 示例演示对输入数据加2的运算
        constexpr int32_t loopCount = 2;
        for (int32_t i = 0; i < loopCount; i++) {
            // 每次循环对输入数据进行加1的运算
            CopyIn();
            Compute();
            CopyOut();
        }
    }
private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        // 对于每个核，除了首次读取外，第二次读取可以命中L2 Cache；
        // 每个核2次读取GM上的数据，2次访问L2 Cache读数据
        DataCopy(xLocal, xGm, BLOCK_LENGTH);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        Adds(yLocal, xLocal, 1, BLOCK_LENGTH);
        inQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        DataCopy(yGm, yLocal, BLOCK_LENGTH);
        inQueueY.FreeTensor(yLocal);
    }
};

extern "C" __global__ __aicore__ void simple_kernel(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm)
{
    AscendC::KernelSample op;
    // 输入数据均等切分成2份数据进行计算
    for (int32_t i = 0; i < TILE_NUM; i++) {
        op.Init(srcGm, dstGm, i);
        op.Process();
    }
}
```
