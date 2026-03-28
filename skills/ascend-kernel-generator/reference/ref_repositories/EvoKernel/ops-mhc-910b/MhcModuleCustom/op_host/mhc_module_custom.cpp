
#include "mhc_module_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static constexpr uint32_t BLOCK_DIM = 32;

static inline uint32_t Ceil8(uint32_t x) { return ((x + 7u) / 8u) * 8u; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MhcModuleCustomTilingData tiling;

    const auto xShape   = context->GetInputShape(0)->GetOriginShape(); // (B,T,N,C)
    const auto scShape  = context->GetInputShape(1)->GetOriginShape(); // (F)
    const auto wpShape  = context->GetInputShape(2)->GetOriginShape(); // (F,N)
    const auto woShape  = context->GetInputShape(3)->GetOriginShape(); // (F,N)
    const auto wrShape  = context->GetInputShape(4)->GetOriginShape(); // (F,NN)
    const auto bpShape  = context->GetInputShape(5)->GetOriginShape(); // (N)
    const auto boShape  = context->GetInputShape(6)->GetOriginShape(); // (N)
    const auto brShape  = context->GetInputShape(7)->GetOriginShape(); // (NN)
    const auto apShape  = context->GetInputShape(8)->GetOriginShape(); // (1)
    const auto aoShape  = context->GetInputShape(9)->GetOriginShape(); // (1)
    const auto arShape  = context->GetInputShape(10)->GetOriginShape();// (1)
    const auto tmShape  = context->GetInputShape(11)->GetOriginShape();// (1)
    const auto epShape  = context->GetInputShape(12)->GetOriginShape();// (1)
    const auto invShape = context->GetInputShape(13)->GetOriginShape();// (1)
    const auto useShape = context->GetInputShape(14)->GetOriginShape();// (1)

    const auto w1Shape  = context->GetInputShape(15)->GetOriginShape(); // (C,H) OR dummy (1,1)
    const auto b1Shape  = context->GetInputShape(16)->GetOriginShape(); // (H)   OR dummy (1)
    const auto w2Shape  = context->GetInputShape(17)->GetOriginShape(); // (H,C) OR dummy (1,1)
    const auto b2Shape  = context->GetInputShape(18)->GetOriginShape(); // (C)   OR dummy (1)
    const auto lnwShape = context->GetInputShape(19)->GetOriginShape(); // (C)   OR dummy (1)
    const auto lnbShape = context->GetInputShape(20)->GetOriginShape(); // (C)   OR dummy (1)
    const auto lneShape = context->GetInputShape(21)->GetOriginShape(); // (1)

    if (xShape.GetDimNum() != 4 || scShape.GetDimNum() != 1 ||
        wpShape.GetDimNum() != 2 || woShape.GetDimNum() != 2 || wrShape.GetDimNum() != 2 ||
        bpShape.GetDimNum() != 1 || boShape.GetDimNum() != 1 || brShape.GetDimNum() != 1 ||
        apShape.GetDimNum() < 1 || aoShape.GetDimNum() < 1 || arShape.GetDimNum() < 1 ||
        tmShape.GetDimNum() < 1 || epShape.GetDimNum() < 1 || invShape.GetDimNum() < 1 ||
        useShape.GetDimNum() < 1 ||
        w1Shape.GetDimNum() != 2 || w2Shape.GetDimNum() != 2 ||
        b1Shape.GetDimNum() != 1 || b2Shape.GetDimNum() != 1 ||
        lnwShape.GetDimNum() != 1 || lnbShape.GetDimNum() != 1 ||
        lneShape.GetDimNum() < 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t B = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t T = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t N = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t C = static_cast<uint32_t>(xShape.GetDim(3));
    if (B == 0u || T == 0u || N == 0u || C == 0u) return ge::GRAPH_FAILED;

    const uint32_t BT = B * T;
    const uint32_t F  = N * C;
    const uint32_t NN = N * N;

    if (static_cast<uint32_t>(scShape.GetDim(0)) != F) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(wpShape.GetDim(0)) != F || static_cast<uint32_t>(wpShape.GetDim(1)) != N) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(woShape.GetDim(0)) != F || static_cast<uint32_t>(woShape.GetDim(1)) != N) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(wrShape.GetDim(0)) != F || static_cast<uint32_t>(wrShape.GetDim(1)) != NN) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(bpShape.GetDim(0)) != N) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(boShape.GetDim(0)) != N) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(brShape.GetDim(0)) != NN) return ge::GRAPH_FAILED;

    // MLP shapes: if real, must align to C/H. If dummy, allow 1s.
    uint32_t H = static_cast<uint32_t>(w1Shape.GetDim(1));
    const uint32_t w1r = static_cast<uint32_t>(w1Shape.GetDim(0));
    const uint32_t w2c = static_cast<uint32_t>(w2Shape.GetDim(1));
    const uint32_t w2r = static_cast<uint32_t>(w2Shape.GetDim(0));
    const uint32_t b1n = static_cast<uint32_t>(b1Shape.GetDim(0));
    const uint32_t b2n = static_cast<uint32_t>(b2Shape.GetDim(0));
    const uint32_t lnwn = static_cast<uint32_t>(lnwShape.GetDim(0));
    const uint32_t lnbn = static_cast<uint32_t>(lnbShape.GetDim(0));

    const bool dummy = (w1r == 1u && H == 1u && w2r == 1u && w2c == 1u && b1n == 1u && b2n == 1u && lnwn == 1u && lnbn == 1u);
    if (!dummy) {
        if (w1r != C) return ge::GRAPH_FAILED;
        if (w2c != C) return ge::GRAPH_FAILED;
        if (w2r != H) return ge::GRAPH_FAILED;
        if (b1n != H) return ge::GRAPH_FAILED;
        if (b2n != C) return ge::GRAPH_FAILED;
        if (lnwn != C) return ge::GRAPH_FAILED;
        if (lnbn != C) return ge::GRAPH_FAILED;
        if (H == 0u) return ge::GRAPH_FAILED;
    } else {
        H = 0u; // not used
    }

    context->SetBlockDim(BLOCK_DIM);

    tiling.set_B(B);
    tiling.set_T(T);
    tiling.set_N(N);
    tiling.set_C(C);
    tiling.set_BT(BT);
    tiling.set_F(F);
    tiling.set_NN(NN);
    tiling.set_Fpad(Ceil8(F));
    tiling.set_NNpad(Ceil8(NN));
    tiling.set_Cpad(Ceil8(C));
    tiling.set_Npad(Ceil8(N));
    tiling.set_H(H);
    tiling.set_Hpad(Ceil8(H));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xs = context->GetInputShape(0);
    if (xs == nullptr || xs->GetDimNum() != 4) return GRAPH_FAILED;

    gert::Shape* out = context->GetOutputShape(0);
    if (out == nullptr) return GRAPH_FAILED;

    // Output is (B,T,N,C)
    out->SetDimNum(4);
    out->SetDim(0, xs->GetDim(0));
    out->SetDim(1, xs->GetDim(1));
    out->SetDim(2, xs->GetDim(2));
    out->SetDim(3, xs->GetDim(3));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class MhcModuleCustom : public OpDef {
public:
    explicit MhcModuleCustom(const char* name) : OpDef(name)
    {
        this->Input("x_streams").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("rms_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("w_pre").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w_post").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w_res").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("b_pre").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_post").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_res").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("alpha_pre").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("alpha_post").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("alpha_res").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("tmax").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("rms_eps").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("invF").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("use_mlp").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("mlp_w1").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mlp_b1").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mlp_w2").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mlp_b2").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("ln_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ln_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ln_eps").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MhcModuleCustom);

} // namespace ops
