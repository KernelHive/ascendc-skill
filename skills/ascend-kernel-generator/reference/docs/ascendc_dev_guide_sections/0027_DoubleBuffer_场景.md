### DoubleBuffer 场景

因存在算子中多次搬入搬出数据的场景，为充分利用硬件资源，实现多流水并行，引入 DoubleBuffer 机制。DoubleBuffer 是通过将输入数据分成大小相等的两块，充分利用 AI Core 的硬件资源，实现数据搬入、计算、数据搬出的并行执行方式。下面以“核间不均分，核内不均分”的样例为例，介绍算子中 DoubleBuffer 的实现，完整样例代码请参见使用 DoubleBuffer 的 Add 算子样例。

![图 6-15 DoubleBuffer 数据切分示意图]()

## Tiling 实现

使能 DoubleBuffer 后，每一个数据块会分成大小相等的两块，因此，若要使能 DoubleBuffer，要求数据总量应该能够均分。为了简化处理，将可用的 Unified Buffer 空间以 32 字节为粒度，分成 n 块 dataBlock，如果 n 不是偶数，则减 1，这样就可以保证一套代码兼容开启或不开启 DoubleBuffer 功能。对应步骤如下：

### 步骤 1

判断数据总长度 totalLength 是否满足 32 字节对齐，如不满足，则计算 totalLength 向上 32 字节对齐后的长度 totalLengthAligned。

```cpp
constexpr uint32_t BLOCK_SIZE = 32;
// 为方便计算，这里根据数据类型定义变量 alignNum 作为对齐数
uint32_t alignNum = BLOCK_SIZE / dataTypeSize;
// totalLength 为数据总量
uint32_t totalLengthAligned = (totalLength % alignNum == 0) ? totalLength : ((totalLength + alignNum - 1) / alignNum) * alignNum;
```

### 步骤 2

根据 totalLengthAligned，计算每个核的计算数据长度 blockLength，分核策略可参照 6.2.4.4 尾核 Tiling。

### 步骤 3

计算其余 Tiling 参数。

对当前 Unified Buffer 可用空间以 32 字节为粒度，进行切分，计算出数据块个数 UB_BLOCK_NUM。根据是否开启 DoubleBuffer 计算出当前可用的最大数据块个数，记作 MAX_AVAILABLE_UB_BLOCK_NUM。最后，以 MAX_AVAILABLE_UB_BLOCK_NUM 为粒度，对 blockLength 进行切分。为方便演示，如下代码直接给出 UB_BLOCK_NUM，作为当前 Unified Buffer 可用空间包含的 block（32 字节）数。

```cpp
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_BLOCK_NUM = 21; // UB 最大可以使用的 block 数量
constexpr uint32_t MAX_AVAILABLE_UB_BLOCK_NUM = UB_BLOCK_NUM / BUFFER_NUM * BUFFER_NUM;

tileNum = blockLength / (alignNum * MAX_AVAILABLE_UB_BLOCK_NUM);
if ((blockLength / alignNum) % MAX_AVAILABLE_UB_BLOCK_NUM == 0 || tileNum == 0) {
    if (tileNum == 0) {
        tileNum = 1;
    }
    if (blockLength < MAX_AVAILABLE_UB_BLOCK_NUM * alignNum) {
        tileLength = ((blockLength / alignNum) + 1) / BUFFER_NUM * BUFFER_NUM * alignNum;
        lastTileLength = tileLength;
    } else {
        tileLength = MAX_AVAILABLE_UB_BLOCK_NUM * alignNum;
        lastTileLength = (blockLength - (tileNum - 1) * tileLength);
    }
} else {
    tileNum = tileNum + 1;
    tileLength = MAX_AVAILABLE_UB_BLOCK_NUM * alignNum;
    lastTileLength = (blockLength - (tileNum - 1) * tileLength);
}
```

## 算子类实现

不开启 DoubleBuffer 时，只需要对每个核上最后一个分块的起始地址做处理；开启 DoubleBuffer 后，需要处理的数据块长度变成原来的一半，所以需要对最后两个数据块的起始地址做处理。

开启 DoubleBuffer，参考 InitBuffer 接口函数原型，将 num 参数配置成 2。

```cpp
pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dataType));
pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(dataType));
pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dataType));
```

同时在计算核内每个数据块的长度时，考虑 DoubleBuffer 场景，需要将 Buffer 数量，即 BUFFER_NUM=2 带入计算。

```cpp
this->tileLength = tiling.tileLength / BUFFER_NUM;
```

由于无法保证尾块满足 DoubleBuffer 的条件，因此不对尾块进行切分。

```cpp
this->lastTileLength = tiling.lastTileLength;
```

Init 函数实现代码如下：

```cpp
__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
{
    if (tiling.isEvenCore) {
        this->blockLength = tiling.blockLength;
        this->tileNum = tiling.tileNum;
        this->tileLength = tiling.tileLength / BUFFER_NUM;
        this->lastTileLength = tiling.lastTileLength;

        xGm.SetGlobalBuffer((__gm__ dataType *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dataType *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ dataType *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
    } else {
        if (AscendC::GetBlockIdx() < tiling.formerNum) {
            this->tileNum = tiling.formerTileNum;
            this->tileLength = tiling.formerTileLength / BUFFER_NUM;
            this->lastTileLength = tiling.formerLastTileLength;

            xGm.SetGlobalBuffer((__gm__ dataType *)x + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
            yGm.SetGlobalBuffer((__gm__ dataType *)y + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
            zGm.SetGlobalBuffer((__gm__ dataType *)z + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
        } else {
            this->tileNum = tiling.tailTileNum;
            this->tileLength = tiling.tailTileLength / BUFFER_NUM;
            this->lastTileLength = tiling.tailLastTileLength;

            xGm.SetGlobalBuffer((__gm__ dataType *)x + tiling.formerLength * tiling.formerNum + tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
            yGm.SetGlobalBuffer((__gm__ dataType *)y + tiling.formerLength * tiling.formerNum + tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
            zGm.SetGlobalBuffer((__gm__ dataType *)z + tiling.formerLength * tiling.formerNum + tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
        }
    }
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dataType));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(dataType));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dataType));
}
```

由于开启 DoubleBuffer 后，切分后的数据块个数翻倍，在 Process 函数中，需要将 BUFFER_NUM 带入计算。

```cpp
__aicore__ inline void Process()
{
    // loop count need to be doubled, due to DoubleBuffer
    constexpr int32_t loopCount = TILE_NUM * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}
```

CopyIn 函数、CopyOut 函数需要对尾块进行单独处理。对于最后一个数据块，先向前偏移 tileLength + lastTileLength 索引，再使用 tileLength 作为单次计算量（仅作为参考，可能并非最佳）。

CopyIn 函数实现代码如下：

```cpp
__aicore__ inline void CopyIn(int32_t progress)
{
    AscendC::LocalTensor<dataType> xLocal = inQueueX.AllocTensor<dataType>();
    AscendC::LocalTensor<dataType> yLocal = inQueueY.AllocTensor<dataType>();
    if (progress == (this->tileNum * BUFFER_NUM - 1)) {
        AscendC::DataCopy(xLocal, xGm[(progress - 2) * this->tileLength + this->lastTileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[(progress - 2) * this->tileLength + this->lastTileLength], this->tileLength);
    } else {
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
    }
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
}
```

CopyOut 函数实现代码如下：

```cpp
__aicore__ inline void CopyOut(int32_t progress)
{
    AscendC::LocalTensor<dataType> zLocal = outQueueZ.DeQue<dataType>();
    if (progress == (this->tileNum * BUFFER_NUM - 1)) {
        AscendC::DataCopy(zGm[(progress - 2) * this->tileLength + this->lastTileLength], zLocal, this->tileLength);
    } else {
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
    }
    outQueueZ.FreeTensor(zLocal);
}
```
