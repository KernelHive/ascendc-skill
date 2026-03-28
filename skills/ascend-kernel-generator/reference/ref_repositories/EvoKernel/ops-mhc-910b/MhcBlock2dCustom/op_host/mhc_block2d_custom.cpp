
#include "mhc_block2d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static constexpr uint32_t BLOCK_DIM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MhcBlock2dCustomTilingData td;

    const auto xShape   = context->GetInputShape(0)->GetOriginShape(); // (B,64,32,32) fp16
    const auto oShape   = context->GetInputShape(1)->GetOriginShape(); // (B,64,32,32) fp16
    const auto mwShape  = context->GetInputShape(2)->GetOriginShape(); // (16,64) fp32
    const auto mbShape  = context->GetInputShape(3)->GetOriginShape(); // (16) fp32

    const auto nsShape  = context->GetInputShape(4)->GetOriginShape(); // (1) i32
    const auto itShape  = context->GetInputShape(5)->GetOriginShape(); // (1) i32
    const auto seShape  = context->GetInputShape(6)->GetOriginShape(); // (1) f32
    const auto stShape  = context->GetInputShape(7)->GetOriginShape(); // (1) f32
    const auto icShape  = context->GetInputShape(8)->GetOriginShape(); // (1) i32
    const auto ocShape  = context->GetInputShape(9)->GetOriginShape(); // (1) i32
    const auto hShape   = context->GetInputShape(10)->GetOriginShape();// (1) i32
    const auto wShape   = context->GetInputShape(11)->GetOriginShape();// (1) i32

    if (xShape.GetDimNum() != 4 || oShape.GetDimNum() != 4) return ge::GRAPH_FAILED;
    if (mwShape.GetDimNum() != 2 || mbShape.GetDimNum() != 1) return ge::GRAPH_FAILED;
    if (nsShape.GetDimNum() < 1 || itShape.GetDimNum() < 1 || seShape.GetDimNum() < 1 || stShape.GetDimNum() < 1 ||
        icShape.GetDimNum() < 1 || ocShape.GetDimNum() < 1 || hShape.GetDimNum() < 1 || wShape.GetDimNum() < 1) return ge::GRAPH_FAILED;

    const uint32_t B = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t C = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t H = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t W = static_cast<uint32_t>(xShape.GetDim(3));
    if (B == 0u || C == 0u || H == 0u || W == 0u) return ge::GRAPH_FAILED;

    // out must match x/out_shape contract for this fused operator
    if ((uint32_t)oShape.GetDim(0) != B || (uint32_t)oShape.GetDim(1) != C ||
        (uint32_t)oShape.GetDim(2) != H || (uint32_t)oShape.GetDim(3) != W) return ge::GRAPH_FAILED;

    // fast path constraints (deepseek-mhc typical block)
    if (C != 64u || H != 32u || W != 32u) return ge::GRAPH_FAILED;
    if ((uint32_t)mwShape.GetDim(0) != 16u || (uint32_t)mwShape.GetDim(1) != 64u) return ge::GRAPH_FAILED;
    if ((uint32_t)mbShape.GetDim(0) != 16u) return ge::GRAPH_FAILED;

    context->SetBlockDim(BLOCK_DIM);

    td.set_B(B);
    td.set_C(C);
    td.set_H(H);
    td.set_W(W);

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
    const gert::Shape* out = context->GetInputShape(1);
    if (out == nullptr || out->GetDimNum() != 4) return GRAPH_FAILED;
    gert::Shape* y = context->GetOutputShape(0);
    *y = *out;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class MhcBlock2dCustom : public OpDef {
public:
    explicit MhcBlock2dCustom(const char* name) : OpDef(name)
    {
        this->Input("x_fp16").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("out_fp16").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("map_w").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("map_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("num_streams").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sinkhorn_iter").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sinkhorn_eps").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sinkhorn_temp").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("in_channels").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("out_channels").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("height").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("width").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y_fp16").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MhcBlock2dCustom);

} // namespace ops
