
#include "parallel_polarized_self_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <algorithm>

namespace optiling {

static constexpr uint32_t BLOCK_SIZE_BYTES = 32; // vector alignment
static constexpr uint32_t MAX_CORE_NUM = 32;     // safe default for 910B; runtime may clamp
static constexpr uint32_t UB_BUDGET_BYTES = 64 * 1024; // conservative UB budget for this op

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ParallelPolarizedSelfAttentionCustomTilingData tiling;

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

    // Validate weights
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

    // Choose core num by positions (each pos is a vector over C)
    uint32_t blockDim = std::min<uint32_t>(MAX_CORE_NUM, totalPos);
    blockDim = std::max<uint32_t>(1, blockDim);
    context->SetBlockDim(blockDim);

    const uint32_t posPerCore = (totalPos + blockDim - 1) / blockDim;

    // cTile selection: fit UB for ping-pong x/cw/y buffers
    // UB use ~ 2*(x + cw + y) = 6*cTile floats + small overhead
    const uint32_t sizeofdt = 4;
    const uint32_t alignElems = BLOCK_SIZE_BYTES / sizeofdt; // 8
    uint32_t maxTileElems = (UB_BUDGET_BYTES / sizeofdt) / 8 * 8; // keep safe
    if (maxTileElems < alignElems) maxTileElems = alignElems;

    // buffers: double-buffer x, cw, y => 6*cTile + a few scalars
    uint32_t cTile = std::min<uint32_t>(C, maxTileElems / 6);
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
class ParallelPolarizedSelfAttentionCustom : public OpDef {
public:
    explicit ParallelPolarizedSelfAttentionCustom(const char* name) : OpDef(name)
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

OP_ADD(ParallelPolarizedSelfAttentionCustom);
} // namespace ops
