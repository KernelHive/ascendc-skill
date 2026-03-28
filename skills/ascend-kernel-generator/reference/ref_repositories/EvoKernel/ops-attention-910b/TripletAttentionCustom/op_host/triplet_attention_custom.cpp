
#include "triplet_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <algorithm>

namespace optiling {
static constexpr uint32_t BLOCK_BYTES = 32; // 32B alignment
static constexpr uint32_t sizeofdatatype = 4; // float32
static constexpr uint32_t ALIGN_ELEMS = BLOCK_BYTES / sizeofdatatype; // 8 floats
static constexpr uint32_t MAX_CORE_NUM_910B = 32;

// UB budget conservative: 4 queues (xch/xcw/xhw/y) with BUFFER_NUM=1
// UB ~= 4 * tileElems * 4B = 16*tileElems bytes. tileElems=8192 => 128KB.
static constexpr uint32_t DEFAULT_TILE_ELEMS = 8192;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TripletAttentionCustomTilingData tiling;

    auto s0 = context->GetInputShape(0)->GetStorageShape();
    auto s1 = context->GetInputShape(1)->GetStorageShape();
    auto s2 = context->GetInputShape(2)->GetStorageShape();

    if (s0.GetDimNum() == 0) return ge::GRAPH_FAILED;
    if (s0.GetDimNum() != s1.GetDimNum() || s0.GetDimNum() != s2.GetDimNum()) return ge::GRAPH_FAILED;

    for (uint32_t i = 0; i < static_cast<uint32_t>(s0.GetDimNum()); ++i) {
        if (s0.GetDim(i) != s1.GetDim(i) || s0.GetDim(i) != s2.GetDim(i)) return ge::GRAPH_FAILED;
        if (s0.GetDim(i) <= 0) return ge::GRAPH_FAILED;
    }

    const uint64_t total64 = s0.GetShapeSize();
    if (total64 == 0 || total64 > UINT32_MAX) return ge::GRAPH_FAILED;
    const uint32_t totalElems = static_cast<uint32_t>(total64);

    // Multi-core split: target ~256KB per core reads (3 inputs) ~ 16K floats/core.
    uint32_t blockDim = (totalElems + 16384 - 1) / 16384;
    blockDim = std::max<uint32_t>(1, blockDim);
    blockDim = std::min<uint32_t>(MAX_CORE_NUM_910B, blockDim);

    // Don't exceed useful blocks (aligned chunks).
    const uint32_t maxUseful = std::max<uint32_t>(1, (totalElems + ALIGN_ELEMS - 1) / ALIGN_ELEMS);
    if (blockDim > maxUseful) blockDim = maxUseful;

    context->SetBlockDim(blockDim);

    // Per-core nominal (ceil), vector part must be ALIGN_ELEMS-aligned.
    uint32_t coreNominal = (totalElems + blockDim - 1) / blockDim;
    uint32_t coreElemsAligned = (coreNominal / ALIGN_ELEMS) * ALIGN_ELEMS;
    if (coreElemsAligned == 0) coreElemsAligned = ALIGN_ELEMS;

    uint32_t tileElems = DEFAULT_TILE_ELEMS;
    tileElems = (tileElems / ALIGN_ELEMS) * ALIGN_ELEMS;
    if (tileElems == 0) tileElems = ALIGN_ELEMS;
    if (tileElems > coreElemsAligned) tileElems = coreElemsAligned;

    tiling.set_totalElems(totalElems);
    tiling.set_blockDim(blockDim);
    tiling.set_coreElemsAligned(coreElemsAligned);
    tiling.set_tileElems(tileElems);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class TripletAttentionCustom : public OpDef {
public:
    explicit TripletAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("x_ch")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x_cw")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x_hw")
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

OP_ADD(TripletAttentionCustom);
} // namespace ops
