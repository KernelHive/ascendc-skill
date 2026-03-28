
#include "static_mhc_hyper_connections_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static constexpr uint32_t BLOCK_DIM = 32;
static constexpr uint32_t TILE_D = 256;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    StaticMhcHyperConnectionsCustomTilingData tiling;

    const auto rShape   = context->GetInputShape(0)->GetOriginShape(); // (B,S,D)
    const auto hrShape  = context->GetInputShape(1)->GetOriginShape(); // (S,S)
    const auto hpShape  = context->GetInputShape(2)->GetOriginShape(); // (S)
    const auto hoShape  = context->GetInputShape(3)->GetOriginShape(); // (S)
    const auto wShape   = context->GetInputShape(4)->GetOriginShape(); // (D,D)
    const auto itShape  = context->GetInputShape(5)->GetOriginShape(); // (1)
    const auto tauShape = context->GetInputShape(6)->GetOriginShape(); // (1)
    const auto lsShape  = context->GetInputShape(7)->GetOriginShape(); // (1)

    if (rShape.GetDimNum() != 3 || hrShape.GetDimNum() != 2 ||
        hpShape.GetDimNum() != 1 || hoShape.GetDimNum() != 1 ||
        wShape.GetDimNum() != 2 ||
        itShape.GetDimNum() < 1 || tauShape.GetDimNum() < 1 || lsShape.GetDimNum() < 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t B = static_cast<uint32_t>(rShape.GetDim(0));
    const uint32_t S = static_cast<uint32_t>(rShape.GetDim(1));
    const uint32_t D = static_cast<uint32_t>(rShape.GetDim(2));
    if (B == 0 || S == 0 || D == 0) return ge::GRAPH_FAILED;

    // Kernel contract: small stream count so we can keep S vectors + SxS in UB easily.
    if (S > 16u) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(hrShape.GetDim(0)) != S || static_cast<uint32_t>(hrShape.GetDim(1)) != S) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(hpShape.GetDim(0)) != S) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(hoShape.GetDim(0)) != S) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(wShape.GetDim(0)) != D || static_cast<uint32_t>(wShape.GetDim(1)) != D) return ge::GRAPH_FAILED;

    context->SetBlockDim(BLOCK_DIM);

    const uint32_t Dpad = ((D + 7u) / 8u) * 8u;

    tiling.set_B(B);
    tiling.set_S(S);
    tiling.set_D(D);
    tiling.set_Dpad(Dpad);
    tiling.set_tileD(TILE_D);

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
    const gert::Shape* r = context->GetInputShape(0);
    if (r == nullptr || r->GetDimNum() != 3) return GRAPH_FAILED;
    gert::Shape* o = context->GetOutputShape(0);
    *o = *r; // (B,S,D)
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class StaticMhcHyperConnectionsCustom : public OpDef {
public:
    explicit StaticMhcHyperConnectionsCustom(const char* name) : OpDef(name)
    {
        this->Input("residuals").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("h_res_logits").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("h_pre_logits").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("h_post_logits").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("branch_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sinkhorn_iters").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("tau").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("log_s").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(StaticMhcHyperConnectionsCustom);

} // namespace ops
