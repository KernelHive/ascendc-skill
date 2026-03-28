
#include "mhc_projector_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

// Keep moderate occupancy; UB grows for N=4 fast path but still fits well.
static constexpr uint32_t BLOCK_DIM = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MhcProjectorCustomTilingData tiling;

    const auto xShape  = context->GetInputShape(0)->GetOriginShape(); // (B,T,N,C)
    const auto ppShape = context->GetInputShape(1)->GetOriginShape(); // (F,N)
    const auto poShape = context->GetInputShape(2)->GetOriginShape(); // (F,N)
    const auto prShape = context->GetInputShape(3)->GetOriginShape(); // (F,NN)
    const auto bpShape = context->GetInputShape(4)->GetOriginShape(); // (N)
    const auto boShape = context->GetInputShape(5)->GetOriginShape(); // (N)
    const auto brShape = context->GetInputShape(6)->GetOriginShape(); // (N,N) or (NN)
    const auto invFShape = context->GetInputShape(12)->GetOriginShape();// (1)

    if (xShape.GetDimNum() != 4 || ppShape.GetDimNum() != 2 || poShape.GetDimNum() != 2 ||
        prShape.GetDimNum() != 2 || bpShape.GetDimNum() != 1 || boShape.GetDimNum() != 1 ||
        (brShape.GetDimNum() != 1 && brShape.GetDimNum() != 2) ||
        invFShape.GetDimNum() < 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t B = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t T = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t N = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t C = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t BT = B * T;
    const uint32_t F  = N * C;
    const uint32_t NN = N * N;

    if (static_cast<uint32_t>(ppShape.GetDim(0)) != F || static_cast<uint32_t>(ppShape.GetDim(1)) != N) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(poShape.GetDim(0)) != F || static_cast<uint32_t>(poShape.GetDim(1)) != N) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(prShape.GetDim(0)) != F || static_cast<uint32_t>(prShape.GetDim(1)) != NN) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(bpShape.GetDim(0)) != N) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(boShape.GetDim(0)) != N) return ge::GRAPH_FAILED;

    if (brShape.GetDimNum() == 2) {
        if (static_cast<uint32_t>(brShape.GetDim(0)) != N || static_cast<uint32_t>(brShape.GetDim(1)) != N) return ge::GRAPH_FAILED;
    } else {
        if (static_cast<uint32_t>(brShape.GetDim(0)) != NN) return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(BLOCK_DIM);

    const uint32_t Fpad  = ((F  + 7u) / 8u) * 8u;
    const uint32_t Npad  = ((N  + 7u) / 8u) * 8u;
    const uint32_t NNpad = ((NN + 7u) / 8u) * 8u;

    tiling.set_B(B);
    tiling.set_T(T);
    tiling.set_N(N);
    tiling.set_C(C);
    tiling.set_BT(BT);
    tiling.set_F(F);
    tiling.set_NN(NN);
    tiling.set_Fpad(Fpad);
    tiling.set_Npad(Npad);
    tiling.set_NNpad(NNpad);

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
    const gert::Shape* xs = context->GetInputShape(0); // (B,T,N,C)
    if (xs == nullptr || xs->GetDimNum() != 4) return GRAPH_FAILED;

    const int64_t B = xs->GetDim(0);
    const int64_t T = xs->GetDim(1);
    const int64_t N = xs->GetDim(2);

    gert::Shape* o0 = context->GetOutputShape(0); // (B,T,N)
    gert::Shape* o1 = context->GetOutputShape(1); // (B,T,N)
    gert::Shape* o2 = context->GetOutputShape(2); // (B,T,N,N)

    o0->SetDimNum(3); o0->SetDim(0, B); o0->SetDim(1, T); o0->SetDim(2, N);
    o1->SetDimNum(3); o1->SetDim(0, B); o1->SetDim(1, T); o1->SetDim(2, N);
    o2->SetDimNum(4); o2->SetDim(0, B); o2->SetDim(1, T); o2->SetDim(2, N); o2->SetDim(3, N);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class MhcProjectorCustom : public OpDef {
public:
    explicit MhcProjectorCustom(const char* name) : OpDef(name)
    {
        this->Input("x_stream").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("phi_pre").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("phi_post").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("phi_res").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_pre").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_post").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_res").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("alpha_pre").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("alpha_post").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("alpha_res").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("tmax").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("rmsnorm_eps").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("invF").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("h_pre").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("h_post").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("h_res").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MhcProjectorCustom);

} // namespace ops
