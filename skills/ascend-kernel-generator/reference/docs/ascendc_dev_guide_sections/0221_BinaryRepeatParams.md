#### BinaryRepeatParams

`BinaryRepeatParams` 是用于控制操作数地址步长的数据结构。结构体内包含操作数相邻迭代间相同 DataBlock 的地址步长、操作数同一迭代内不同 DataBlock 的地址步长等参数。

- 相邻迭代间的地址步长参数说明请参考 `repeatStride`；
- 同一迭代内 DataBlock 的地址步长参数说明请参考 `dataBlockStride`。

结构体具体定义为：

```cpp
const int32_t DEFAULT_BLK_NUM = 8;
const int32_t DEFAULT_BLK_STRIDE = 1;
const uint8_t DEFAULT_REPEAT_STRIDE = 8;

struct BinaryRepeatParams {
    __aicore__ BinaryRepeatParams() {}
    __aicore__ BinaryRepeatParams(
        const uint8_t dstBlkStrideIn,
        const uint8_t src0BlkStrideIn,
        const uint8_t src1BlkStrideIn,
        const uint8_t dstRepStrideIn,
        const uint8_t src0RepStrideIn,
        const uint8_t src1RepStrideIn
    ) : dstBlkStride(dstBlkStrideIn),
        src0BlkStride(src0BlkStrideIn),
        src1BlkStride(src1BlkStrideIn),
        dstRepStride(dstRepStrideIn),
        src0RepStride(src0RepStrideIn),
        src1RepStride(src1RepStrideIn)
    {}

    uint32_t blockNumber = DEFAULT_BLK_NUM;
    uint8_t dstBlkStride = DEFAULT_BLK_STRIDE;
    uint8_t src0BlkStride = DEFAULT_BLK_STRIDE;
    uint8_t src1BlkStride = DEFAULT_BLK_STRIDE;
    uint8_t dstRepStride = DEFAULT_REPEAT_STRIDE;
    uint8_t src0RepStride = DEFAULT_REPEAT_STRIDE;
    uint8_t src1RepStride = DEFAULT_REPEAT_STRIDE;
    bool repeatStrideMode = false;
    bool strideSizeMode = false;
};
```

其中：

- `blockNumber`、`repeatStrideMode` 和 `strideSizeMode` 为保留参数，用户无需关心，使用默认值即可。
- 用户需要自行定义 DataBlock Stride 参数，包含 `dstBlkStride`、`src0BlkStride` 和 `src1BlkStride`，以及 Repeat Stride 参数，包含 `dstRepStride`、`src0RepStride` 和 `src1RepStride`。
