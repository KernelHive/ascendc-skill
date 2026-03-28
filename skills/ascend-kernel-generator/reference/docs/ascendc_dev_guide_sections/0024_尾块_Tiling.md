#### 尾块 Tiling

## 场景说明

### 对齐场景

如下图中的示例，算子的输入 shape 为 (1, 2048)，支持的数据类型为 half 类型，输入数据可以对齐到一个 datablock 的大小（32 字节），输入数据为 2048 * 2 / 32 = 128 个 datablock，因此可以平均分配到每个核上（假设使用 8 个核），每个核上处理 256 个数，16 个 datablock。此时不需要进行尾块处理。

图 6-11 shape 对齐场景

### 尾块场景

针对一些 shape，比如算子的输入 shape 为 (1, 1904)，支持的数据类型为 half 类型，输入数据可以对齐到一个 datablock 的大小（32 字节），可以平均分配到每个核上（假设使用 8 个核），每个核上处理 238 个数，238 个数无法均分到 datablock 上，分满 14 个 datablock 后，剩余 14 个数（28 字节），多核切分后需要进行尾块处理。

对于不同 shape 的输入进行数据切分时，可能会发生 Tiling 后的数据平均分配到多核上，但每个核内的数据无法均分的情况。针对此种场景，在 Tiling 参数中增加变量 `lastTileLength`，用来表示最后一个分块，即尾块的大小。

## Tiling 结构体

在定义算子的 Tiling 结构体时包含以下四个成员：

- `blockLength`：每个核上计算的数据长度；
- `tileNum`：每个核上切分的数据块的个数；
- `tileLength`：每个核上除尾块外，每个数据块的长度；
- `lastTileLength`：每个核上尾块的长度。

其中，当 `lastTileLength` 等于 `tileLength` 时，为核间均分同时核内均分场景，因此这两种场景可以做代码归一化处理。

图 6-12 多核 Tiling 尾块示意图

## Tiling 实现

### 结构体定义

算子的 Tiling 结构体定义如下：

```cpp
struct AddCustomTilingData {
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
};
```

### Host 侧实现步骤

Host 侧 Tiling 实现的主要内容为计算以上四个成员变量。步骤如下：

#### 步骤 1：数据对齐

判断数据总长度 `totalLength` 是否满足 32 字节对齐，如不满足，则计算 `totalLength` 向上 32 字节对齐后的长度 `totalLengthAligned`。

```cpp
constexpr uint32_t BLOCK_SIZE = 32;
// 为方便计算，这里根据数据类型定义变量 alignNum 作为对齐数
uint32_t alignNum = BLOCK_SIZE / dataTypeSize;
// totalLength 为数据总量
uint32_t totalLengthAligned = (totalLength % alignNum == 0) ?
    totalLength : ((totalLength + alignNum - 1) / alignNum) * alignNum;
```

#### 步骤 2：核间均分

判断 `totalLengthAligned` 是否能被使用的核数 `BLOCK_DIM` 均分，如果可以，则计算每个核上计算数据长度 `blockLength`。

```cpp
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t UB_BLOCK_NUM = 20; // 此处为方便验证，使用 UB_BLOCK_NUM 作为 Unified Buffer 可用的 Block 数量，因此可得出可用 UB 空间的大小为 UB_BLOCK_NUM * BLOCK_SIZE
uint32_t blockLength, tileNum;
if (totalLengthAligned % BLOCK_DIM == 0) {
    blockLength = totalLengthAligned / BLOCK_DIM;
}
```

#### 步骤 3：计算 tileNum

为了减少数据搬运开销，应尽量使用核内的 Unified Buffer 空间。基于每个核上的计算量以及可用 Unified Buffer 空间的大小，计算 `tileNum`。

```cpp
tileNum = blockLength / alignNum / UB_BLOCK_NUM;
```

#### 步骤 4：计算 tileLength 和 lastTileLength

根据计算出的 `tileNum`，计算 `tileLength` 和 `lastTileLength`。

如果每个核的计算量能够被当前可用 Unified Buffer 空间均分，或者计算量小于可用 Unified Buffer 空间，则按照无尾块场景处理。

```cpp
// (blockLength / alignNum) % UB_BLOCK_NUM 为 0，表示每个核的计算量能够被当前可用 Unified Buffer 空间均分
// tileNum 为 0，表示计算量小于可用 Unified Buffer 空间
if ((blockLength / alignNum) % UB_BLOCK_NUM == 0 || tileNum == 0) {
    if (tileNum == 0) {
        tileNum = 1;
    }
    if (blockLength < UB_BLOCK_NUM * alignNum) {
        tileLength = ((blockLength + alignNum - 1) / alignNum) * alignNum;
        lastTileLength = tileLength;
    } else {
        tileLength = UB_BLOCK_NUM * alignNum;
        lastTileLength = tileLength;
    }
}
```

反之，按照尾块场景处理，在 `tileNum` 上加 1 作为每个核的数据块个数，尾块长度为单核计算数据长度 - (tileNum - 1) * tileLength。

```cpp
else {
    tileNum = tileNum + 1;
    tileLength = UB_BLOCK_NUM * alignNum;
    lastTileLength = blockLength - (tileNum - 1) * tileLength;
}
```

### Host 侧完整代码

```cpp
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t UB_BLOCK_NUM = 20; // 此处为方便验证，使用 UB_BLOCK_NUM 作为 UB 可用的 Block 数量，因此可得出可用 UB 空间的大小为 UB_BLOCK_NUM * BLOCK_SIZE
...

uint32_t alignNum = BLOCK_SIZE / dataTypeSize; // 为方便计算，这里根据数据类型定义变量 alignNum 作为对齐数，dataTypeSize 为运算数据的数据类型对应的字节数
// totalLength 为数据总量
uint32_t totalLengthAligned = (totalLength % alignNum == 0) ?
    totalLength : ((totalLength + alignNum - 1) / alignNum) * alignNum;
uint32_t blockLength, tileNum;
if (totalLengthAligned % BLOCK_DIM == 0) {
    blockLength = totalLengthAligned / BLOCK_DIM;
    tileNum = blockLength / alignNum / UB_BLOCK_NUM;
    if ((blockLength / alignNum) % UB_BLOCK_NUM == 0 || tileNum == 0) {
        if (tileNum == 0) {
            tileNum = 1;
        }
        if (blockLength < UB_BLOCK_NUM * alignNum) {
            tileLength = ((blockLength + alignNum - 1) / alignNum) * alignNum;
            lastTileLength = tileLength;
        } else {
            tileLength = UB_BLOCK_NUM * alignNum;
            lastTileLength = tileLength;
        }
    } else {
        tileNum = tileNum + 1;
        tileLength = UB_BLOCK_NUM * alignNum;
        lastTileLength = blockLength - (tileNum - 1) * tileLength;
    }
    ...
}
```

## 算子类实现

由于尾块长度为 `lastTileLength`，与其它数据块的长度不同，因此 `CopyIn` 函数、`CopyOut` 函数需要对尾块单独处理。

### CopyIn 函数

```cpp
__aicore__ inline void CopyIn(int32_t progress)
{
    AscendC::LocalTensor<dataType> xLocal = inQueueX.AllocTensor<dataType>();
    AscendC::LocalTensor<dataType> yLocal = inQueueY.AllocTensor<dataType>();
    if (progress == (this->tileNum - 1)) {
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength],
            this->lastTileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength],
            this->lastTileLength);
    } else {
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
    }
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
}
```

### CopyOut 函数

```cpp
__aicore__ inline void CopyOut(int32_t progress)
{
    AscendC::LocalTensor<dataType> zLocal = outQueueZ.DeQue<dataType>();
    if (progress == (this->tileNum - 1)) {
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal,
            this->lastTileLength);
    } else {
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
    }
    outQueueZ.FreeTensor(zLocal);
}
```
