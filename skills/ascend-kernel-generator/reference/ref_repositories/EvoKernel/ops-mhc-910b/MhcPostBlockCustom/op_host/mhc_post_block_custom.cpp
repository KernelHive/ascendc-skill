
#include "mhc_post_block_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Slightly higher to improve occupancy; kernel is UB-heavy but within limits.
static constexpr uint32_t BLOCK_DIM = 28;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MhcPostBlockCustomTilingData td;

    const auto xShape = context->GetInputShape(0)->GetOriginShape(); // (N,H) fp16
    const auto rShape = context->GetInputShape(1)->GetOriginShape(); // (N,S,H) fp16
    const auto pShape = context->GetInputShape(2)->GetOriginShape(); // (N,S,1) fp32
    const auto cShape = context->GetInputShape(3)->GetOriginShape(); // (N,S,S) fp32

    const auto hsShape = context->GetInputShape(4)->GetOriginShape(); // (1) i32
    const auto hmShape = context->GetInputShape(5)->GetOriginShape(); // (1) i32

    if (xShape.GetDimNum() != 2 || rShape.GetDimNum() != 3 ||
        pShape.GetDimNum() != 3 || cShape.GetDimNum() != 3) return ge::GRAPH_FAILED;
    if (hsShape.GetDimNum() < 1 || hmShape.GetDimNum() < 1) return ge::GRAPH_FAILED;

    const uint32_t N = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t H = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t S = static_cast<uint32_t>(rShape.GetDim(1));

    if (N == 0u || H == 0u || S == 0u) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(rShape.GetDim(0)) != N || static_cast<uint32_t>(rShape.GetDim(2)) != H) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(pShape.GetDim(0)) != N || static_cast<uint32_t>(pShape.GetDim(1)) != S || static_cast<uint32_t>(pShape.GetDim(2)) != 1u) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(cShape.GetDim(0)) != N || static_cast<uint32_t>(cShape.GetDim(1)) != S || static_cast<uint32_t>(cShape.GetDim(2)) != S) return ge::GRAPH_FAILED;

    if (S != 4u) return ge::GRAPH_FAILED;

    context->SetBlockDim(BLOCK_DIM);

    td.set_N(N);
    td.set_S(S);
    td.set_H(H);

    td.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(td.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* r = context->GetInputShape(1); // residual (N,S,H)
    if (r == nullptr || r->GetDimNum() != 3) return GRAPH_FAILED;

    gert::Shape* out = context->GetOutputShape(0); // out (N,S,H)
    out->SetDimNum(3);
    out->SetDim(0, r->GetDim(0));
    out->SetDim(1, r->GetDim(1));
    out->SetDim(2, r->GetDim(2));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class MhcPostBlockCustom : public OpDef {
public:
    explicit MhcPostBlockCustom(const char* name) : OpDef(name)
    {
        this->Input("x_fp16").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("residual_fp16").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("post_layer_mix").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("comb_res_mix").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("hidden_size").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("hc_mult").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("out_fp16").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MhcPostBlockCustom);

} // namespace ops
