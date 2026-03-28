
#include "criss_cross_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <algorithm>

namespace optiling {
static constexpr uint32_t BLOCK_BYTES = 32; // 32B alignment for GM/UB vector ops
static constexpr uint32_t sizeofdatatype = 4; // float32
static constexpr uint32_t ALIGN_ELEMS = BLOCK_BYTES / sizeofdatatype; // 8 floats
static constexpr uint32_t MAX_CORE_NUM_910B = 32;

// Keep UB conservative and aligned. With BUFFER_NUM=1 and 4 queues (x/oh/ow/y):
// UB ~= 4 * tileElems * 4B = 16*tileElems bytes. tileElems=8192 => 128KB.
static constexpr uint32_t DEFAULT_TILE_ELEMS = 8192;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CrissCrossAttentionCustomTilingData tiling;

    auto x_shape  = context->GetInputShape(0)->GetStorageShape();
    auto oh_shape = context->GetInputShape(1)->GetStorageShape();
    auto ow_shape = context->GetInputShape(2)->GetStorageShape();
    auto g_shape  = context->GetInputShape(3)->GetStorageShape();

    if (x_shape.GetDimNum() == 0) return ge::GRAPH_FAILED;
    if (x_shape.GetDimNum() != oh_shape.GetDimNum() || x_shape.GetDimNum() != ow_shape.GetDimNum()) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < static_cast<uint32_t>(x_shape.GetDimNum()); ++i) {
        if (x_shape.GetDim(i) != oh_shape.GetDim(i) || x_shape.GetDim(i) != ow_shape.GetDim(i)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (g_shape.GetShapeSize() != 1) return ge::GRAPH_FAILED;

    const uint64_t total64 = x_shape.GetShapeSize();
    if (total64 == 0 || total64 > UINT32_MAX) return ge::GRAPH_FAILED;
    const uint32_t totalElems = static_cast<uint32_t>(total64);

    // Multi-core split: target ~256KB per core of reads (3 inputs) => about 16K floats/core.
    // Clamp to [1, 32].
    uint32_t blockDim = (totalElems + 16384 - 1) / 16384;
    blockDim = std::max<uint32_t>(1, blockDim);
    blockDim = std::min<uint32_t>(MAX_CORE_NUM_910B, blockDim);

    // Don't exceed available alignment blocks.
    const uint32_t maxUseful = std::max<uint32_t>(1, (totalElems + ALIGN_ELEMS - 1) / ALIGN_ELEMS);
    if (blockDim > maxUseful) blockDim = maxUseful;

    context->SetBlockDim(blockDim);

    // Per-core nominal range (ceil-div), but vectorizable part must be ALIGN_ELEMS-aligned.
    uint32_t coreNominal = (totalElems + blockDim - 1) / blockDim;
    uint32_t coreElemsAligned = (coreNominal / ALIGN_ELEMS) * ALIGN_ELEMS; // floor to aligned
    if (coreElemsAligned == 0) coreElemsAligned = ALIGN_ELEMS; // ensure progress; tail handled scalar

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
class CrissCrossAttentionCustom : public OpDef {
public:
    explicit CrissCrossAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("out_h")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("out_w")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma")
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

OP_ADD(CrissCrossAttentionCustom);
} // namespace ops
