
#include <cstdint>
#include "matrix_scalar_multiplication_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

static inline uint64_t CeilDivU64(uint64_t a, uint64_t b) { return (a + b - 1ULL) / b; }

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    auto aDesc = context->GetInputTensor(0);
    auto sDesc = context->GetInputTensor(1);
    if (aDesc == nullptr || sDesc == nullptr) return ge::GRAPH_FAILED;

    // Flatten A (any-rank ND) into total element count.
    auto aShape = aDesc->GetOriginShape();
    const int32_t dimNum = aShape.GetDimNum();
    if (dimNum <= 0) return ge::GRAPH_FAILED;

    uint64_t total = 1ULL;
    for (int32_t i = 0; i < dimNum; ++i) {
        const int64_t d = aShape.GetDim(i);
        if (d <= 0) return ge::GRAPH_FAILED;
        total *= static_cast<uint64_t>(d);
        if (total == 0ULL) return ge::GRAPH_FAILED;
    }

    // Scalar S must have exactly 1 element (shape arbitrary ND with numel==1).
    auto sShape = sDesc->GetOriginShape();
    const int32_t sDimNum = sShape.GetDimNum();
    if (sDimNum <= 0) return ge::GRAPH_FAILED;

    uint64_t sTotal = 1ULL;
    for (int32_t i = 0; i < sDimNum; ++i) {
        const int64_t d = sShape.GetDim(i);
        if (d <= 0) return ge::GRAPH_FAILED;
        sTotal *= static_cast<uint64_t>(d);
    }
    if (sTotal != 1ULL) return ge::GRAPH_FAILED;

    // Conservative UB tile size (float in/out): 2 * tileLen * 4 bytes.
    // 8192 => ~64KB (+overhead), typically safe.
    constexpr uint32_t kTileLen = 8192U;

    MatrixScalarMultiplicationCustomTilingData td;
    td.set_totalLength(total);
    td.set_tileLength(kTileLen);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNum());
    if (coreNum == 0U) coreNum = 1U;

    uint32_t tileNum = static_cast<uint32_t>(CeilDivU64(total, static_cast<uint64_t>(kTileLen)));
    if (tileNum == 0U) tileNum = 1U;

    uint32_t blockDim = (tileNum < coreNum) ? tileNum : coreNum;
    if (blockDim == 0U) blockDim = 1U;
    context->SetBlockDim(blockDim);

    // Single kernel path.
    context->SetTilingKey(0);

    td.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(td.GetDataSize());

    // No workspace needed.
    size_t *ws = context->GetWorkspaceSizes(1);
    if (ws == nullptr) return ge::GRAPH_FAILED;
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class MatrixScalarMultiplicationCustom : public OpDef {
public:
    explicit MatrixScalarMultiplicationCustom(const char *name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        // Scalar as a 1-element ND tensor on NPU.
        this->Input("s")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(MatrixScalarMultiplicationCustom);

} // namespace ops
