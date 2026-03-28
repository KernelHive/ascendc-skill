### 限制 TilingData 结构大小

**优先级**：中

**描述**  
TilingData 结构是 Tiling 切分信息的载体。当 Host 侧按照 Tiling 切分策略计算完 Tiling 后，算子会以入参的方式将 Tiling 切分信息从 Host 侧传递到 Device 侧，此时 Tiling 信息存放在 GM 上。调用 `GET_TILING_DATA` 宏后，会将 Tiling 信息从 GM 拷贝到 AI 处理器的栈空间上，期间会有拷贝开销。由于 GM 访问效率较低，同时考虑到栈空间限制，需要限制 TilingData 结构大小。拷贝耗时为 us 级别，在小 shape 的场景下，进行此类优化收益会更加明显。

限制 TilingData 结构大小，可以从以下方面考虑：

- 减少不必要的 TilingData 结构变量；
- 根据 Tiling 的数据范围选择合适的变量类型；
- 合理排布 TilingData 结构；
- TilingData 整体结构要求 8 字节补齐。

---

## 反例

### 变量冗余与类型不合理

以下示例中存在 TilingData 结构变量冗余的情况：`BlockDim` 信息已经通过 `SetBlockDim` 接口进行设置，可以在 Kernel 侧调用 `GetBlockNum` 接口获取，无需通过 TilingData 结构传递。

此外，变量的数据类型也不合理：
- `formerNum` 和 `tailNum` 分别为计算整块数据的核数和计算尾块数据的核数，不会超过 `BLOCK_DIM` 的值，使用 `uint8_t` 类型即可；
- `formerLength` 等变量根据其计算逻辑，不会超出 `uint32_t` 的范围，使用 `uint32_t` 类型即可。

```c
// Tiling结构体定义
BEGIN_TILING_DATA_DEF(TilingDataUnalign)
TILING_DATA_FIELD_DEF(uint64_t, blockDim);
TILING_DATA_FIELD_DEF(uint64_t, formerNum);
TILING_DATA_FIELD_DEF(uint64_t, tailNum);
TILING_DATA_FIELD_DEF(uint64_t, formerLength);
TILING_DATA_FIELD_DEF(uint64_t, tailLength);
TILING_DATA_FIELD_DEF(uint64_t, alignNum);
END_TILING_DATA_DEF;

// Host侧Tiling函数计算Tiling结构信息
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t SIZE_OF_HALF = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t ALIGN_NUM = BLOCK_SIZE / SIZE_OF_HALF;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingDataUnalign tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    // BlockDim信息已经通过SetBlockDim接口进行设置
    context->SetBlockDim(BLOCK_DIM);
    uint32_t totalLengthAligned = ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    // formerNum、tailNum保证不超过0-BLOCK_DIM数据范围
    uint32_t formerNum = (totalLengthAligned / ALIGN_NUM) % BLOCK_DIM;
    uint32_t tailNum = BLOCK_DIM - formerNum;
    // formerLength等变量根据其计算逻辑，不会超出uint32_t的范围
    uint32_t formerLength = ((totalLengthAligned / BLOCK_DIM + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    uint32_t tailLength = (totalLengthAligned / BLOCK_DIM / ALIGN_NUM) * ALIGN_NUM;
    ...
}
```

---

## 正例

Tiling 变量无冗余，变量数据类型最小化。

```c
BEGIN_TILING_DATA_DEF(TilingDataUnalign)
TILING_DATA_FIELD_DEF(uint8_t, formerNum);
TILING_DATA_FIELD_DEF(uint8_t, tailNum);
TILING_DATA_FIELD_DEF(uint32_t, formerLength);
TILING_DATA_FIELD_DEF(uint32_t, tailLength);
TILING_DATA_FIELD_DEF(uint32_t, alignNum);
END_TILING_DATA_DEF;
```

---

## 反例

### 结构排布不合理

以下示例中 TilingData 结构不合理：由于 AI 处理器访存需要 8 字节对齐，在用户定义 TilingData 结构后，Ascend C 工程框架会按照 8 字节对齐的方式对字节进行补齐，并保证整体 TilingData 结构满足 8 字节对齐要求。如下 TilingData 结构中 `formerNum` 和 `tailNum` 变量都会补充 3 个字节，整体 TilingData 结构会因为 8 字节对齐再补充 4 个字节，该 TilingData 结构共计补充 10 个字节。

```c
BEGIN_TILING_DATA_DEF(TilingDataUnalign)
TILING_DATA_FIELD_DEF(uint8_t, formerNum);  // 需补充3个字节，使得formerLength变量访问无误
TILING_DATA_FIELD_DEF(uint32_t, formerLength);
TILING_DATA_FIELD_DEF(uint8_t, tailNum);    // 需补充3个字节，使得tailLength变量访问无误
TILING_DATA_FIELD_DEF(uint32_t, tailLength);
TILING_DATA_FIELD_DEF(uint32_t, alignNum);  // 需补充4个字节，使得下个TilingData结构访问无误
END_TILING_DATA_DEF;
```

---

## 正例

### 结构排布优化

以下示例中，对 Tiling 参数的排布进行了调整，字节排布合理，只需要补充 2 个字节，达到了减小 TilingData 结构的目的。

```c
BEGIN_TILING_DATA_DEF(TilingDataUnalign)
TILING_DATA_FIELD_DEF(uint8_t, formerNum);
TILING_DATA_FIELD_DEF(uint8_t, tailNum);    // 需补充2个字节，使得formerLength变量访问无误
TILING_DATA_FIELD_DEF(uint32_t, formerLength);
TILING_DATA_FIELD_DEF(uint32_t, tailLength);
TILING_DATA_FIELD_DEF(uint32_t, alignNum);
END_TILING_DATA_DEF;
```
