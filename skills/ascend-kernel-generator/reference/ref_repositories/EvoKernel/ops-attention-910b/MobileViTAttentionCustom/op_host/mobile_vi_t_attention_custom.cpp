
#include "mobile_vi_t_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static constexpr uint32_t BLOCK_BYTES = 32;
static constexpr uint32_t FLOAT_BYTES = 4;
static constexpr uint32_t ALIGN_ELEMS = BLOCK_BYTES / FLOAT_BYTES; // 8 floats

// Kernel uses 3 UB tiles: outPing + outPong + tmp.
// Conservative budget to leave headroom for compiler/runtime.
static constexpr uint32_t UB_BUDGET_BYTES = 120 * 1024;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MobileViTAttentionCustomTilingData tiling;

    const auto& s0 = context->GetInputShape(0)->GetStorageShape();
    const auto& s1 = context->GetInputShape(1)->GetStorageShape();
    const auto& s2 = context->GetInputShape(2)->GetStorageShape();

    if (s0.GetDimNum() == 0) return ge::GRAPH_FAILED;
    if (s0.GetDimNum() != s1.GetDimNum() || s0.GetDimNum() != s2.GetDimNum()) return ge::GRAPH_FAILED;
    for (uint32_t i = 0; i < static_cast<uint32_t>(s0.GetDimNum()); ++i) {
        if (s0.GetDim(i) != s1.GetDim(i) || s0.GetDim(i) != s2.GetDim(i)) return ge::GRAPH_FAILED;
    }

    const uint64_t total64 = s0.GetShapeSize();
    if (total64 == 0 || total64 > UINT32_MAX) return ge::GRAPH_FAILED;

    const uint32_t totalElems = static_cast<uint32_t>(total64);
    const uint32_t totalAligned = ((totalElems + ALIGN_ELEMS - 1) / ALIGN_ELEMS) * ALIGN_ELEMS;

    // 3 tiles in UB.
    uint32_t maxTileElems = UB_BUDGET_BYTES / (3 * FLOAT_BYTES);
    maxTileElems = (maxTileElems / ALIGN_ELEMS) * ALIGN_ELEMS;
    if (maxTileElems < ALIGN_ELEMS) maxTileElems = ALIGN_ELEMS;
    if (maxTileElems > totalAligned) maxTileElems = totalAligned;

    // Keep a reasonably large tile to amortize overhead; cap at maxTileElems.
    uint32_t tileElems = maxTileElems;
    if (tileElems < 4096) tileElems = std::min<uint32_t>(4096, maxTileElems);
    tileElems = (tileElems / ALIGN_ELEMS) * ALIGN_ELEMS;
    if (tileElems == 0) tileElems = ALIGN_ELEMS;

    const uint32_t tileNum = (totalAligned + tileElems - 1) / tileElems;

    tiling.set_totalElems(totalElems);
    tiling.set_totalAligned(totalAligned);
    tiling.set_tileElems(tileElems);
    tiling.set_tileNum(tileNum);

    uint32_t block_dim = std::min<uint32_t>(tileNum, 48);
    if (block_dim == 0) block_dim = 1;
    context->SetBlockDim(block_dim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) return ge::GRAPH_FAILED;
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
class MobileViTAttentionCustom : public OpDef {
public:
    explicit MobileViTAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("att_delta").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ffn_delta").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MobileViTAttentionCustom);
} // namespace ops
