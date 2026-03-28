
#include "hard_tanh_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t x, uint32_t y) { return (x + y - 1u) / y; }

static constexpr uint32_t kDefaultBlockDim = 48;     // good occupancy without too many tiny chunks
static constexpr uint32_t kMinElemsPerBlock = 8192;  // avoid too-small per-core slices

// HardTanh is simple; use large tiles to reduce queue overhead and MTE setup cost.
static constexpr uint32_t kTileCandidates[] = {65536, 32768, 16384, 8192, 4096};

// Conservative UB budget for double-buffered x/y:
// 2*(x+y) => 4*tile*4B = 16*tile bytes.
static constexpr uint32_t kUbBudgetBytes = 192 * 1024;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    HardTanhCustomTilingData tiling;

    const uint32_t totalLength =
        static_cast<uint32_t>(context->GetInputShape(0)->GetOriginShape().GetShapeSize());

    uint32_t blockDim = kDefaultBlockDim;
    if (totalLength == 0) blockDim = 1;
    blockDim = std::max(1u, std::min(blockDim, CeilDivU32(std::max(1u, totalLength), kMinElemsPerBlock)));
    context->SetBlockDim(blockDim);

    // Per-core contiguous chunk length (ceil), choose tile so each core has a few tiles if possible.
    const uint32_t perCore = CeilDivU32(std::max(1u, totalLength), blockDim);

    uint32_t tile = kTileCandidates[2]; // default 16384
    if (perCore >= kTileCandidates[0] * 2) tile = kTileCandidates[0];
    else if (perCore >= kTileCandidates[1] * 2) tile = kTileCandidates[1];
    else if (perCore >= kTileCandidates[2] * 2) tile = kTileCandidates[2];
    else if (perCore >= kTileCandidates[3] * 2) tile = kTileCandidates[3];
    else tile = kTileCandidates[4];

    // UB constraint for double-buffered x/y.
    // Need ~= 16*tile bytes.
    if (16u * tile > kUbBudgetBytes) {
        // Fall back down the candidate list.
        for (uint32_t i = 0; i < sizeof(kTileCandidates)/sizeof(kTileCandidates[0]); ++i) {
            uint32_t cand = kTileCandidates[i];
            if (16u * cand <= kUbBudgetBytes) { tile = cand; break; }
        }
        if (16u * tile > kUbBudgetBytes) tile = 2048; // final safety fallback
    }

    if (totalLength != 0 && tile > totalLength) tile = totalLength;
    if (tile == 0) tile = 1;

    tiling.set_totalLength(totalLength);
    tiling.set_blockDim(blockDim);
    tiling.set_tileSize(tile);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto dt = context->GetInputDataType(0);
    context->SetOutputDataType(0, dt);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class HardTanhCustom : public OpDef {
public:
    explicit HardTanhCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(HardTanhCustom);

}  // namespace ops
