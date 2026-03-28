
#include "dense_sparse_attention_custom_tiling.h"
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
    auto ix_shape = context->GetInputShape(2)->GetStorageShape();

    if (q_shape.GetDimNum() != 4 || kv_shape.GetDimNum() != 4 || ix_shape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t B   = static_cast<uint32_t>(q_shape.GetDim(0));
    const uint32_t Sq  = static_cast<uint32_t>(q_shape.GetDim(1));
    const uint32_t H   = static_cast<uint32_t>(q_shape.GetDim(2));
    const uint32_t Dqk = static_cast<uint32_t>(q_shape.GetDim(3));

    const uint32_t NB  = static_cast<uint32_t>(kv_shape.GetDim(0));
    const uint32_t PBS = static_cast<uint32_t>(kv_shape.GetDim(1));
    const uint32_t one = static_cast<uint32_t>(kv_shape.GetDim(2));
    const uint32_t Dkv = static_cast<uint32_t>(kv_shape.GetDim(3));

    const uint32_t B2   = static_cast<uint32_t>(ix_shape.GetDim(0));
    const uint32_t Sq2  = static_cast<uint32_t>(ix_shape.GetDim(1));
    const uint32_t topk = static_cast<uint32_t>(ix_shape.GetDim(2));

    if (B == 0u || Sq == 0u || H == 0u || Dqk == 0u) return ge::GRAPH_FAILED;
    if (NB == 0u || PBS == 0u) return ge::GRAPH_FAILED;
    if (one != 1u) return ge::GRAPH_FAILED;
    if (Dkv != Dqk) return ge::GRAPH_FAILED;
    if (B2 != B || Sq2 != Sq) return ge::GRAPH_FAILED;
    if (topk == 0u) return ge::GRAPH_FAILED;

    if (Sq != 1u) return ge::GRAPH_FAILED;
    if (Dqk != 576u) return ge::GRAPH_FAILED;
    if (PBS != 16u) return ge::GRAPH_FAILED;

    const uint32_t Dv = 512u;
    if (Dv > Dqk) return ge::GRAPH_FAILED;
    if (topk > 32u) return ge::GRAPH_FAILED;

    const uint32_t flatKV = NB * PBS;
    if (flatKV == 0u) return ge::GRAPH_FAILED;

    DenseSparseAttentionCustomTilingData tiling;
    tiling.set_B(B);
    tiling.set_Sq(Sq);
    tiling.set_H(H);
    tiling.set_Dqk(Dqk);
    tiling.set_Dv(Dv);
    tiling.set_NB(NB);
    tiling.set_PBS(PBS);
    tiling.set_topk(topk);
    tiling.set_flatKV(flatKV);
    tiling.set_totalTasks(B * H);
    tiling.set_scale(1.0f / std::sqrt(static_cast<float>(Dqk)));

    const uint32_t totalTasks = B * H;
    uint32_t coreNum = std::min<uint32_t>(std::max<uint32_t>(1u, totalTasks), 24u);
    tiling.set_coreNum(coreNum);
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
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class DenseSparseAttentionCustom : public OpDef {
public:
    explicit DenseSparseAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("kv_cache").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(DenseSparseAttentionCustom);

} // namespace ops
