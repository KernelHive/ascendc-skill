
#include "paged_attention_kv_cache_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    // q: [B, Sq, Hq, D] bf16
    // k_cache/v_cache: [NB, PBS, Hkv, D] bf16
    // cache_seqlens: [B] int32
    // page_table: [B, MBS] int32
    // causal_flag: [1] int32
    auto q_shape  = context->GetInputShape(0)->GetStorageShape();
    auto k_shape  = context->GetInputShape(1)->GetStorageShape();
    auto v_shape  = context->GetInputShape(2)->GetStorageShape();
    auto sl_shape = context->GetInputShape(3)->GetStorageShape();
    auto pt_shape = context->GetInputShape(4)->GetStorageShape();
    auto cf_shape = context->GetInputShape(5)->GetStorageShape();

    if (q_shape.GetDimNum() != 4 || k_shape.GetDimNum() != 4 || v_shape.GetDimNum() != 4) return ge::GRAPH_FAILED;
    if (sl_shape.GetDimNum() != 1 || pt_shape.GetDimNum() != 2 || cf_shape.GetDimNum() != 1) return ge::GRAPH_FAILED;

    const uint32_t B  = static_cast<uint32_t>(q_shape.GetDim(0));
    const uint32_t Sq = static_cast<uint32_t>(q_shape.GetDim(1));
    const uint32_t Hq = static_cast<uint32_t>(q_shape.GetDim(2));
    const uint32_t D  = static_cast<uint32_t>(q_shape.GetDim(3));

    const uint32_t NB  = static_cast<uint32_t>(k_shape.GetDim(0));
    const uint32_t PBS = static_cast<uint32_t>(k_shape.GetDim(1));
    const uint32_t Hkv = static_cast<uint32_t>(k_shape.GetDim(2));
    const uint32_t Dk  = static_cast<uint32_t>(k_shape.GetDim(3));

    const uint32_t Bpt = static_cast<uint32_t>(pt_shape.GetDim(0));
    const uint32_t MBS = static_cast<uint32_t>(pt_shape.GetDim(1));

    if (B == 0 || Sq == 0 || Hq == 0 || D == 0 || NB == 0 || PBS == 0 || Hkv == 0 || MBS == 0) return ge::GRAPH_FAILED;
    if (Dk != D) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(v_shape.GetDim(0)) != NB ||
        static_cast<uint32_t>(v_shape.GetDim(1)) != PBS ||
        static_cast<uint32_t>(v_shape.GetDim(2)) != Hkv ||
        static_cast<uint32_t>(v_shape.GetDim(3)) != D) return ge::GRAPH_FAILED;

    if (Bpt != B) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(sl_shape.GetDim(0)) != B) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(cf_shape.GetDim(0)) != 1u) return ge::GRAPH_FAILED;

    // GQA constraint
    if ((Hq % Hkv) != 0u) return ge::GRAPH_FAILED;

    PagedAttentionKVCacheCustomTilingData tiling;
    tiling.set_B(B);
    tiling.set_Sq(Sq);
    tiling.set_Hq(Hq);
    tiling.set_D(D);
    tiling.set_NB(NB);
    tiling.set_PBS(PBS);
    tiling.set_Hkv(Hkv);
    tiling.set_MBS(MBS);
    tiling.set_maxSeq(MBS * PBS);
    tiling.set_groups(Hq / Hkv);
    tiling.set_totalTasks(B * Hq);
    tiling.set_scale(1.0f / std::sqrt(static_cast<float>(D)));

    const uint32_t totalTasks = B * Hq;
    uint32_t coreNum = std::min<uint32_t>(std::max<uint32_t>(1u, totalTasks), 24u);
    context->SetBlockDim(coreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    auto* outShape = context->GetOutputShape(0);
    const auto* qShape = context->GetInputShape(0);
    if (outShape == nullptr || qShape == nullptr) return GRAPH_FAILED;
    *outShape = *qShape; // out same shape as q: [B,Sq,Hq,D]
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0)); // bf16
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {

class PagedAttentionKVCacheCustom : public OpDef {
public:
    explicit PagedAttentionKVCacheCustom(const char* name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k_cache").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v_cache").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cache_seqlens").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("page_table").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("causal_flag").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(PagedAttentionKVCacheCustom);

} // namespace ops
