#### 多核 Tiling

基于 Ascend C 方式实现带有 Tiling 的算子的开发流程如下图所示。

![算子开发流程](图6-7)

## 算子分析

本样例为输入数据在核间均分、核内均分场景。本样例的 Tiling 策略为：数据整体长度 `TOTAL_LENGTH` 为 `8 * 2048`，数据平均分配到 8 个核上运行，每个核上计算的数据长度 `BLOCK_LENGTH` 为 `2048`，将单核上的数据切分成 16 块（此处切分成 16 块仅用来作为 Tiling 的样例，并不代表性能最佳，仅供参考），每块数据的长度 `TILE_LENGTH` 为 `128`。数据切分示意如下图所示：

![数据切分示意图](图6-8)

### Ascend C Add 算子设计规格

| 属性 | 值 |
|------|-----|
| 算子类型 (OpType) | Add |

| 输入/输出 | name | shape | data type | format |
|-----------|------|-------|-----------|--------|
| 输入 | x | (8, 2048) | half | ND |
| 输入 | y | (8, 2048) | half | ND |
| 输出 | z | (8, 2048) | half | ND |

| 项目 | 内容 |
|------|------|
| 核函数名称 | `add_custom` |
| 使用的主要接口 | `DataCopy`：数据搬移接口<br>`Add`：矢量基础算术接口<br>`EnQue`、`DeQue`等接口：Queue队列管理接口 |
| 算子实现文件名称 | `add_custom.cpp` |

## Tiling 实现

前述场景中算子的输入和输出均为固定 shape，然而在实际的算子开发场景中，这些信息是支持动态变化的，场景会更加灵活和复杂。动态 shape 场景下，输入的 shape 是未知的。一些与输入 shape 相关的变量（比如每次搬运的块大小等），需要通过 Tiling 计算出来，然后传递到 kernel 侧，kernel 侧使用该参数进行后续的计算。

具体实现方式为：分析设计 Tiling 参数、定义 Tiling 结构体，在 HOST 侧通过上下文获取输入输出的 shape 信息，根据 shape 信息，计算 Tiling 参数并设置到对应的 Tiling 结构体中；通过核函数入口参数将 Tiling 信息传入核函数，在核函数内通过解析 Tiling 结构体，获取并使用相关参数来实现核函数内部逻辑，详细介绍请参考 Host 侧 tiling 实现。

本节将以上述分析中的切分策略为例，说明如何实现 Tiling。

基于本节的切分策略，Tiling 需要定义如下参数：

- `blockLength`：每个核的计算数据长度；
- `tileNum`：每个核需要计算的数据块个数；
- `tileLength`：每个核内每个数据块的长度。

根据确定的 Tiling 参数，在算子 TilingData 结构定义头文件中，使用 C++ 语法定义 `TilingData` 结构体，代码如下。该头文件命名为“算子名称_tiling.h”。本章节中的算子名称为 `add_custom`，对应头文件命名应为 `add_custom_tiling.h`。

```cpp
struct AddCustomTilingData {
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};
```

接下来，创建一个与 Tiling 结构体头文件对应的 cpp 文件 `add_custom_tiling.cpp`，并在该文件内完成 Tiling 参数的计算。由于每个核内数据被切分为 16 块，根据使用的核数和核内切分数，计算 Tiling 参数，并写入到 Tiling 结构体内。代码示例如下：

```cpp
#include "add_custom_tiling.h"

constexpr int32_t CORE_NUM = 8; // 使用的核数
constexpr int32_t TILE_NUM = 16; // 核内切分数量

void GenerateTilingData(uint8_t* tilingBuf)
{
    uint32_t totalLength;
    // 此处省略如何获取数据总长 TOTAL_LENGTH，可以根据具体情况实现。本章节仅介绍 Tiling 相关内容。
    AddCustomTilingData *tiling = reinterpret_cast<AddCustomTilingData *>(tilingBuf);
    uint32_t blockLength = TOTAL_LENGTH / CORE_NUM;
    uint32_t tileNum = TILE_NUM;
    uint32_t tileLength = blockLength / tileNum;

    tiling->blockLength = blockLength;
    tiling->tileNum = tileNum;
    tiling->tileLength = tileLength;
}
```

最后，在 Host 侧调用程序中，调用上述 Tiling 参数计算函数，计算出相关参数，然后传递到 Kernel 侧核函数。

```cpp
extern void GenerateTilingData(uint8_t* tilingBuf);

constexpr int32_t CORE_NUM = 8;
...
uint8_t *tiling = nullptr;

size_t tilingSize = sizeof(AddCustomTilingData);
#ifdef ASCENDC_CPU_DEBUG
    tiling = (uint8_t *)AscendC::GmAlloc(tilingSize); // CPU Debug模式
    ...
#else
    ...
    CHECK_ACL(aclrtMallocHost((void **)(&tiling), tilingSize)); // NPU模式
    ...
#endif

GenerateTilingData(tiling); // 调用 tiling 参数计算函数
...

#ifdef ASCENDC_CPU_DEBUG
    ...
    ICPU_RUN_KF(add_custom, CORE_NUM, x, y, z,
                *reinterpret_cast<AddCustomTilingData *>(tiling)); // CPU Debug模式下核函数调用
    ...
#else
    ...
    ACLRT_LAUNCH_KERNEL(add_custom)(CORE_NUM, stream, xDevice, yDevice, zDevice, // NPU模式下核函数调用
                                    reinterpret_cast<AddCustomTilingData *>(tiling));
    ...
#endif
```

## 算子类实现

Kernel 侧算子实现仍遵循矢量算子核函数实现流程，接下来重点介绍本场景中算子类实现的不同点。

### 设置输入输出 Global Tensor 的 Global Memory 内存地址

由于本样例中将数据分配到了多个核上进行处理，每个核处理不同的数据，因此不同核要处理的数据在 Global Memory 上的地址不同，在初始化函数 `Init` 中，需要获取单核所需处理的输入输出在 Global Memory 上的内存偏移地址，并将该偏移地址设置到 `GlobalTensor` 中。

以获取输入 `x` 在 Global Memory 上的内存偏移地址为例，数据整体长度 `TOTAL_LENGTH` 为 `8 * 2048`，平均分配到 8 个核上运行，每个核上处理的数据长度 `blockLength` 为 `2048`，调用 `GetBlockIdx` 接口获取当前核的 index，`x + blockLength * GetBlockIdx()` 即为单核处理程序中 `x` 在 Global Memory 上的内存偏移地址，获取偏移地址后，使用 `GlobalTensor` 类的 `SetGlobalBuffer` 接口设定该核上 Global Memory 的起始地址以及长度，具体示意图请参考图 6-9。代码如下所示：

```cpp
xGm.SetGlobalBuffer((__gm__ half *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
```

![多核并行处理示意图](图6-9)

### 通过 Pipe 内存管理对象为输入输出 Queue 分配内存

对于单核上的处理数据，可以进行数据切块（Tiling），在本示例中，仅作为参考，将单核上的数据（2048 个数）切分成 16 块（并不意味着 16 块就是性能最优），每块 `tileLength`（128）个数据。数据切分示意图如图 6-10 所示。

![单核数据切分示意图](图6-10)

与基础矢量算子相比，在通过 Pipe 内存管理对象为输入输出 Queue 分配内存时，需使用单核内每个数据块的长度 `tileLength` 作为分配内存的长度。比如，为输入 `x` 的 Queue 分配内存，可以通过如下代码段实现，Pipe 为 `inQueueX` 分配了一块大小为 `tileLength * sizeof(half)` 个字节的内存块，每个内存块能容纳 `tileLength`（128）个 half 类型数据。

```cpp
pipe.InitBuffer(inQueueX, 1, this->tileLength * sizeof(half));
```

具体的初始化函数代码如下：

```cpp
__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
{
    this->blockLength = tiling.blockLength;
    this->tileNum = tiling.tileNum;
    this->tileLength = tiling.tileLength;

    // 计算每个核上的地址偏移
    xGm.SetGlobalBuffer((__gm__ half *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
    yGm.SetGlobalBuffer((__gm__ half *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
    zGm.SetGlobalBuffer((__gm__ half *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueueX, 1, this->tileLength * sizeof(half));
    pipe.InitBuffer(inQueueY, 1, this->tileLength * sizeof(half));
    pipe.InitBuffer(outQueueZ, 1, this->tileLength * sizeof(half));
}
```

每个核需要对 `tileNum` 个数据块分别进行搬入、计算、搬出处理，因此 `Process` 函数内将 `tileNum` 作为循环上限。

```cpp
__aicore__ inline void Process()
{
    int32_t loopCount = this->tileNum;
    // tiling strategy, pipeline parallel
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}
```

对应的，每个核内搬入、搬出每个数据块时，需定位到每个数据块所在 Global Memory 上的内存偏移地址，因此在 `CopyIn` 和 `CopyOut` 函数内部使用 `DataCopy` 接口时，需增加每个数据块的地址偏移。`Compute` 函数没有变化，与基础矢量算子相同。

`CopyIn` 函数实现代码如下：

```cpp
__aicore__ inline void CopyIn(int32_t progress)
{
    ...
    // copy progress_th tile from global tensor to local tensor
    AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
    AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
    ...
}
```

`CopyOut` 函数实现代码如下：

```cpp
__aicore__ inline void CopyOut(int32_t progress)
{
    ...
    // copy progress_th tile from local tensor to global tensor
    AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
    ...
}
```

## 运行验证

Host 侧的核函数调用程序，实现从 Host 侧的 APP 程序调用算子，进行运行验证。在程序中调用开启多核运行的核函数时，需要指定使用的核数，代码如下所示。

### CPU 侧运行验证

```cpp
constexpr uint32_t BLOCK_DIM = 8;
...
// 调用 ICPU_RUN_KF 调测宏，完成核函数 CPU 侧的调用
ICPU_RUN_KF(add_custom, BLOCK_DIM, x, y, z, *reinterpret_cast<AddCustomTilingData *>(tiling));
// 输出数据写出
...
```

### NPU 侧运行验证

```cpp
constexpr uint32_t BLOCK_DIM = 8;
...
// 用 ACLRT_LAUNCH_KERNEL 接口调用核函数完成指定的运算
ACLRT_LAUNCH_KERNEL(add_custom)(BLOCK_DIM, stream, xDevice, yDevice, zDevice,
                                *reinterpret_cast<AddCustomTilingData *>(tiling));
// 用内核调用符 <<<>>> 调用核函数完成指定的运算，add_custom_do 中封装了 <<<>>> 调用
...
```
