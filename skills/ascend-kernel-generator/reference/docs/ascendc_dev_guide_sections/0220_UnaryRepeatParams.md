#### UnaryRepeatParams

`UnaryRepeatParams` 是用于控制操作数地址步长的数据结构。结构体内包含操作数相邻迭代间相同 DataBlock 的地址步长、操作数同一迭代内不同 DataBlock 的地址步长等参数。

- 相邻迭代间的地址步长参数说明请参考 `repeatStride`；
- 同一迭代内 DataBlock 的地址步长参数说明请参考 `dataBlockStride`。

结构体具体定义为：

```cpp
const int32_t DEFAULT_BLK_NUM = 8;
const int32_t DEFAULT_BLK_STRIDE = 1;
const uint8_t DEFAULT_REPEAT_STRIDE = 8;

struct UnaryRepeatParams {
    __aicore__ UnaryRepeatParams() {}
    __aicore__ UnaryRepeatParams(const uint16_t dstBlkStrideIn, const uint16_t srcBlkStrideIn,
                                 const uint8_t dstRepStrideIn, const uint8_t srcRepStrideIn)
        : dstBlkStride(dstBlkStrideIn),
          srcBlkStride(srcBlkStrideIn),
          dstRepStride(dstRepStrideIn),
          srcRepStride(srcRepStrideIn)
    {}
    __aicore__ UnaryRepeatParams(const uint16_t dstBlkStrideIn, const uint16_t srcBlkStrideIn,
                                 const uint8_t dstRepStrideIn, const uint8_t srcRepStrideIn, const bool halfBlockIn)
        : dstBlkStride(dstBlkStrideIn),
          srcBlkStride(srcBlkStrideIn),
          dstRepStride(dstRepStrideIn),
          srcRepStride(srcRepStrideIn),
          halfBlock(halfBlockIn)
    {}

    uint32_t blockNumber = DEFAULT_BLK_NUM;
    uint16_t dstBlkStride = DEFAULT_BLK_STRIDE;
    uint16_t srcBlkStride = DEFAULT_BLK_STRIDE;
    uint8_t dstRepStride = DEFAULT_REPEAT_STRIDE;
    uint8_t srcRepStride = DEFAULT_REPEAT_STRIDE;
    bool repeatStrideMode = false;
    bool strideSizeMode = false;
    bool halfBlock = false;
};
```

其中：

- `blockNumber`、`repeatStrideMode`、`strideSizeMode` 为保留参数，用户无需关心，使用默认值即可。
- `halfBlock` 表示 `CastDeq` 指令的结果写入对应 UB 的上半（`halfBlock = true`）还是下半（`halfBlock = false`）部分。

用户需要自行定义以下参数：

- **DataBlock Stride 参数**：包含 `dstBlkStride`、`srcBlkStride`；
- **Repeat Stride 参数**：包含 `dstRepStride`、`srcRepStride`。
