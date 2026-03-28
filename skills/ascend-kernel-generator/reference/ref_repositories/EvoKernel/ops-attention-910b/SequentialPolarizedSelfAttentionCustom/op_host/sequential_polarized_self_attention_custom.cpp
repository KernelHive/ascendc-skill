
#include "sequential_polarized_self_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <algorithm>

namespace optiling {

static constexpr uint32_t BLOCK_SIZE_BYTES = 32; // 32B alignment
static constexpr uint32_t MAX_CORE_NUM = 40;     // 910B typically supports many AICores; keep conservative-ish
static constexpr uint32_t UB_BUDGET_BYTES = 96 * 1024; // allow bigger channel tile while staying safe

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SequentialPolarizedSelfAttentionCustomTilingData tiling;

    auto xs  = context->GetInputShape(0)->GetStorageShape(); // x
    auto cws = context->GetInputShape(1)->GetStorageShape(); // channel_weight
    auto sws = context->GetInputShape(2)->GetStorageShape(); // spatial_weight

    if (xs.GetDimNum() != 4 || cws.GetDimNum() != 4 || sws.GetDimNum() != 4) return ge::GRAPH_FAILED;

    const uint64_t B64 = xs.GetDim(0);
    const uint64_t C64 = xs.GetDim(1);
    const uint64_t H64 = xs.GetDim(2);
    const uint64_t W64 = xs.GetDim(3);
    if (B64 == 0 || C64 == 0 || H64 == 0 || W64 == 0) return ge::GRAPH_FAILED;
    if (B64 > UINT32_MAX || C64 > UINT32_MAX || H64 > UINT32_MAX || W64 > UINT32_MAX) return ge::GRAPH_FAILED;

    const uint32_t B = static_cast<uint32_t>(B64);
    const uint32_t C = static_cast<uint32_t>(C64);
    const uint32_t H = static_cast<uint32_t>(H64);
    const uint32_t W = static_cast<uint32_t>(W64);

    // Validate weights:
    if ((uint32_t)cws.GetDim(0) != B || (uint32_t)cws.GetDim(1) != C ||
        (uint32_t)cws.GetDim(2) != 1 || (uint32_t)cws.GetDim(3) != 1) return ge::GRAPH_FAILED;

    if ((uint32_t)sws.GetDim(0) != B || (uint32_t)sws.GetDim(1) != 1 ||
        (uint32_t)sws.GetDim(2) != H || (uint32_t)sws.GetDim(3) != W) return ge::GRAPH_FAILED;

    const uint64_t HW64 = (uint64_t)H * (uint64_t)W;
    if (HW64 == 0 || HW64 > UINT32_MAX) return ge::GRAPH_FAILED;
    const uint32_t HW = static_cast<uint32_t>(HW64);

    const uint64_t totalPos64 = (uint64_t)B * (uint64_t)HW;
    if (totalPos64 == 0 || totalPos64 > UINT32_MAX) return ge::GRAPH_FAILED;
    const uint32_t totalPos = static_cast<uint32_t>(totalPos64);

    // Pick cores based on totalPos; prefer more cores to reduce tail and improve overlap.
    uint32_t blockDim = std::min<uint32_t>(MAX_CORE_NUM, totalPos);
    blockDim = std::max<uint32_t>(1, blockDim);
    context->SetBlockDim(blockDim);

    const uint32_t posPerCore = (totalPos + blockDim - 1) / blockDim;

    // UB sizing:
    // We keep three UB tensors: xTile, cwTile, swVec, gate, yTile.
    // But to reduce UB footprint, we reuse buffers: swVec shares with gate, etc.
    // Budget in floats:
    const uint32_t sizeofdt = 4;
    const uint32_t alignElems = BLOCK_SIZE_BYTES / sizeofdt; // 8 floats
    uint32_t maxElems = (UB_BUDGET_BYTES / sizeofdt);
    maxElems = (maxElems / alignElems) * alignElems;
    if (maxElems < alignElems) maxElems = alignElems;

    // Conservative: 4*cTile floats needed simultaneously (x,cw,swVec,y) with reuse.
    uint32_t cTile = std::min<uint32_t>(C, maxElems / 4);
    cTile = (cTile / alignElems) * alignElems;
    if (cTile == 0) cTile = std::min<uint32_t>(C, alignElems);

    tiling.set_B(B);
    tiling.set_C(C);
    tiling.set_H(H);
    tiling.set_W(W);
    tiling.set_HW(HW);
    tiling.set_totalPos(totalPos);
    tiling.set_blockDim(blockDim);
    tiling.set_posPerCore(posPerCore);
    tiling.set_cTile(cTile);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
class SequentialPolarizedSelfAttentionCustom : public OpDef {
public:
    explicit SequentialPolarizedSelfAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("channel_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("spatial_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SequentialPolarizedSelfAttentionCustom);
} // namespace ops
