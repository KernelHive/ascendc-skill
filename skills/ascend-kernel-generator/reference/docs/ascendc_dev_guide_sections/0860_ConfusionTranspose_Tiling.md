##### ConfusionTranspose Tiling

## 类型与位置

类型为 `LocalTensor`，支持的 `TPosition` 为 `VECIN`/`VECCALC`/`VECOUT`。

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| `transposeTy` | 输入 | 数据排布及 reshape 的类型，类型为 `TransposeType` 枚举类。 |
| `tiling` | 输入 | 计算所需 tiling 信息，Tiling 信息的获取请参考 [ConfusionTranspose Tiling](#confusiontranspose-tiling)。 |

### TransposeType 枚举

```cpp
enum class TransposeType : uint8_t {
    TRANSPOSE_TYPE_NONE,           // default value
    TRANSPOSE_NZ2ND_0213,          // 场景1：NZ2ND，1、2轴互换
    TRANSPOSE_NZ2NZ_0213,          // 场景2：NZ2NZ，1、2轴互换
    TRANSPOSE_NZ2NZ_012_WITH_N,    // 场景3：NZ2NZ，尾轴切分
    TRANSPOSE_NZ2ND_012_WITH_N,    // 场景4：NZ2ND，尾轴切分
    TRANSPOSE_NZ2ND_012_WITHOUT_N, // 场景5：NZ2ND，尾轴合并
    TRANSPOSE_NZ2NZ_012_WITHOUT_N, // 场景6：NZ2NZ，尾轴合并
    TRANSPOSE_ND2ND_ONLY,          // 场景7：二维转置
    TRANSPOSE_ND_UB_GM,            // 当前不支持
    TRANSPOSE_GRAD_ND_UB_GM,       // 当前不支持
    TRANSPOSE_ND2ND_B16,           // 当前不支持
    TRANSPOSE_NCHW2NHWC,           // 当前不支持
    TRANSPOSE_NHWC2NCHW            // 当前不支持
};
```

## 返回值说明

无

## 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束。

## 调用示例

本示例为场景1（NZ2ND，1、2轴互换）示例：

- 输入 Tensor：
  - `shape: [B, N, H/N/16, S/16, 16, 16]`
  - `origin_shape: [B, N, S, H/N]`
  - `format: "NZ"`
  - `origin_format: "ND"`
- 输出 Tensor：
  - `shape: [B, S, N, H/N]`
  - `origin_shape: [B, S, N, H/N]`
  - `format: "ND"`
  - `origin_format: "ND"`

参数：`B=1`，`N=2`，`S=64`，`H/N=32`，输入数据类型均为 `half`。

更多算子样例请参考 ConfusionTranspose 算子样例。

```cpp
AscendC::TPipe *pipe = pipeIn;
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrcVecIn;
AscendC::TQue<AscendC::TPosition::VECOUT, 1> inQueueSrcVecOut;
pipe->InitBuffer(inQueueSrcVecIn, 1, b * n * s * hnDiv * sizeof(T));
pipe->InitBuffer(inQueueSrcVecOut, 1, b * n * s * hnDiv * sizeof(T));
AscendC::ConfusionTranspose(dstLocal, srcLocal, AscendC::TransposeType::TRANSPOSE_NZ2ND_0213, this->tiling);
```

## ConfusionTranspose Tiling

### 功能说明

用于获取 ConfusionTranspose Tiling 参数。

### 函数原型

```cpp
void GetConfusionTransposeMaxMinTmpSize(
    const ge::Shape &srcShape,
    const uint32_t typeSize,
    const uint32_t transposeTypeIn,
    uint32_t &maxValue,
    uint32_t &minValue
)

void GetConfusionTransposeTilingInfo(
    const ge::Shape &srcShape,
    const uint32_t stackBufferSize,
    const uint32_t typeSize,
    const uint32_t transposeTypeIn,
    optiling::ConfusionTransposeTiling &tiling
)
```

### 参数说明

#### GetConfusionTransposeMaxMinTmpSize 接口参数说明

| 参数名 | 输入/输出 | 含义 |
|--------|-----------|------|
| `srcShape` | 输入 | 输入 Tensor 的 shape 信息，具体 `srcShape` 传入格式为：<br>场景1：[B, N, S, H/N]<br>场景2：[B, N, S, H/N]<br>场景3：[B, N, S, H/N]<br>场景4：[B, N, S, H/N]<br>场景5：[B, N, S, H/N]<br>场景6：[B, N, S, H/N]<br>场景7：[H, W] |
| `typeSize` | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为 `half`，此处应传入 2。 |
| `transposeTypeIn` | 输入 | 选择数据排布及 reshape 的类型，根据输入数字选择对应的场景，参数范围为 [1,7]。<br>场景1（NZ2ND，1、2轴互换）：1<br>场景2（NZ2NZ，1、2轴互换）：2<br>场景3（NZ2NZ，尾轴切分）：3<br>场景4（NZ2ND，尾轴切分）：4<br>场景5（NZ2ND，尾轴合并）：5<br>场景6（NZ2NZ，尾轴合并）：6<br>场景7（二维转置）：7 |
| `maxValue` | 输出 | ConfusionTranspose 接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel 侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。<br>**说明**：`maxValue` 仅作为参考值，有可能大于 Unified Buffer 剩余空间的大小，该场景下，开发者需要根据 Unified Buffer 剩余空间的大小来选取合适的临时空间大小。 |
| `minValue` | 输出 | ConfusionTranspose 接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。 |

#### GetConfusionTransposeTilingInfo 接口参数列表

| 参数名 | 输入/输出 | 含义 |
|--------|-----------|------|
| `srcShape` | 输入 | 输入的 shape 信息，具体 `srcShape` 传入格式为：<br>场景1：[B, N, S, H/N]<br>场景2：[B, N, S, H/N]<br>场景3：[B, N, S, H/N]<br>场景4：[B, N, S, H/N]<br>场景5：[B, N, S, H/N]<br>场景6：[B, N, S, H/N]<br>场景7：[H, W] |
| `stackBufferSize` | 输入 | 可供 ConfusionTranspose 接口计算的空间大小，单位 Byte。 |
| `typeSize` | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为 `half`，此处应传入 2。 |
| `transposeTypeIn` | 输入 | 选择数据排布及 reshape 的类型，根据输入数字选择对应的场景，参数范围为 [1,7]。<br>场景1（NZ2ND，1、2轴互换）：1<br>场景2（NZ2NZ，1、2轴互换）：2<br>场景3（NZ2NZ，尾轴切分）：3<br>场景4（NZ2ND，尾轴切分）：4<br>场景5（NZ2ND，尾轴合并）：5<br>场景6（NZ2NZ，尾轴合并）：6<br>场景7（二维转置）：7 |
| `tiling` | 输出 | 输入数据的切分信息。 |

### 返回值说明

无

### 约束说明

无

### 调用示例

如下样例介绍了使用 ConfusionTranspose 高阶 API 时 host 侧获取 Tiling 参数的流程以及该参数如何在 kernel 侧使用。样例中为场景1，输入 Tensor 的 shape 大小为 `[1, 2, 64, 32]`，输入的数据类型为 `half`。

#### 步骤1：将 ConfusionTransposeTiling 结构体参数增加至 TilingData 结构体

```cpp
BEGIN_TILING_DATA_DEF(TilingData) // 注册一个 tiling 的类，以 tiling 的名字作为入参
    TILING_DATA_FIELD_DEF(uint32_t, tileNum); // 添加 tiling 字段，每个核上总计算数据分块个数
    ... // 添加其他 tiling 字段
    TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingData); // 将 ConfusionTransposeTiling 结构体参数增加至 TilingData 结构体
END_TILING_DATA_DEF;
```

#### 步骤2：Tiling 实现函数中获取 ConfusionTranspose kernel 侧接口所需 tiling 参数

```cpp
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_tileNum(TILE_NUM);
    // 设置其他 Tiling 参数
    ...
    std::vector<int64_t> shapeVec = {1, 2, 64, 32};
    ge::Shape srcShape(shapeVec);

    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetConfusionTransposeMaxMinTmpSize(srcShape, sizeof(half), maxValue, minValue);
    // 本样例中仅作为样例说明，获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
    const uint32_t stackBufferSize = minValue;
    // 获取 ConfusionTranspose Tiling 参数
    AscendC::GetConfusionTransposeTilingInfo(srcShape, stackBufferSize, sizeof(half), 1, tiling.confusionTransposeTilingData);
    ... // 其他逻辑
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

#### 步骤3：kernel 侧获取 TilingData 并传入 ConfusionTranspose 接口

完整的 kernel 侧样例请参考 [ConfusionTranspose](#confusiontranspose)。

```cpp
extern "C" __global__ __aicore__ void func_custom(GM_ADDR src_gm, GM_ADDR dst_gm, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(TilingData, tiling);

    KernelConfusionTranspose<half> op;
    op.Init(src_gm, dst_gm, TilingData.confusionTransposeTilingData);
    op.Process();
}
```
