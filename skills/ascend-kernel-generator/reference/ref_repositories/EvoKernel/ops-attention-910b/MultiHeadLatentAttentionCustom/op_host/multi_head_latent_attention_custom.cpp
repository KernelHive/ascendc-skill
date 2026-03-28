
#include "multi_head_latent_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>
#include <cmath>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    auto q_shape  = context->GetInputShape(0)->GetStorageShape();
    auto kv_shape = context->GetInputShape(1)->GetStorageShape();
    auto bt_shape = context->GetInputShape(2)->GetStorageShape();
    auto sl_shape = context->GetInputShape(3)->GetStorageShape();
    auto cf_shape = context->GetInputShape(4)->GetStorageShape();

    if (q_shape.GetDimNum() != 4 || kv_shape.GetDimNum() != 4) return ge::GRAPH_FAILED;
    if (bt_shape.GetDimNum() != 2 || sl_shape.GetDimNum() != 1 || cf_shape.GetDimNum() != 1) return ge::GRAPH_FAILED;

    const uint32_t B   = static_cast<uint32_t>(q_shape.GetDim(0));
    const uint32_t Sq  = static_cast<uint32_t>(q_shape.GetDim(1));
    const uint32_t Hq  = static_cast<uint32_t>(q_shape.GetDim(2));
    const uint32_t Dqk = static_cast<uint32_t>(q_shape.GetDim(3));

    const uint32_t NB  = static_cast<uint32_t>(kv_shape.GetDim(0));
    const uint32_t PBS = static_cast<uint32_t>(kv_shape.GetDim(1));
    const uint32_t mid = static_cast<uint32_t>(kv_shape.GetDim(2));
    const uint32_t Dkv = static_cast<uint32_t>(kv_shape.GetDim(3));

    const uint32_t Bbt = static_cast<uint32_t>(bt_shape.GetDim(0));
    const uint32_t MBS = static_cast<uint32_t>(bt_shape.GetDim(1));

    if (B == 0 || Sq == 0 || Hq == 0 || Dqk == 0) return ge::GRAPH_FAILED;
    if (NB == 0 || PBS == 0 || MBS == 0) return ge::GRAPH_FAILED;
    if (mid != 1u) return ge::GRAPH_FAILED;
    if (Dkv != Dqk) return ge::GRAPH_FAILED;
    if (Bbt != B) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(sl_shape.GetDim(0)) != B) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(cf_shape.GetDim(0)) != 1u) return ge::GRAPH_FAILED;

    // Decode-only specialization.
    if (Sq != 1u) return ge::GRAPH_FAILED;
    if (Dqk != 576u) return ge::GRAPH_FAILED;
    if (PBS != 16u) return ge::GRAPH_FAILED;

    const uint32_t Dv = 512u;
    if (Dv > Dqk) return ge::GRAPH_FAILED;

    MultiHeadLatentAttentionCustomTilingData tiling;
    tiling.set_B(B);
    tiling.set_Sq(Sq);
    tiling.set_Hq(Hq);
    tiling.set_Dqk(Dqk);
    tiling.set_Dv(Dv);
    tiling.set_NB(NB);
    tiling.set_PBS(PBS);
    tiling.set_MBS(MBS);
    tiling.set_maxSeq(MBS * PBS);
    tiling.set_totalTasks(B * Hq);
    tiling.set_scale(1.0f / std::sqrt(static_cast<float>(Dqk)));

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
    if (context == nullptr) return GRAPH_FAILED;
    const auto* qShape = context->GetInputShape(0);
    auto* outShape = context->GetOutputShape(0);
    if (qShape == nullptr || outShape == nullptr) return GRAPH_FAILED;

    *outShape = *qShape;
    outShape->SetDimNum(4);
    outShape->SetDim(0, qShape->GetDim(0));
    outShape->SetDim(1, qShape->GetDim(1));
    outShape->SetDim(2, qShape->GetDim(2));
    outShape->SetDim(3, 512);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0)); // bf16
    return GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class MultiHeadLatentAttentionCustom : public OpDef {
public:
    explicit MultiHeadLatentAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("kv_cache").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("block_table").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cache_seqlens").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("causal_flag").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MultiHeadLatentAttentionCustom);

} // namespace ops
