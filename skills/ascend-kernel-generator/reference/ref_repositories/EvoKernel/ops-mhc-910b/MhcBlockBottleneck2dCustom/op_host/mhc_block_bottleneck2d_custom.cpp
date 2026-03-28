
#include "mhc_block_bottleneck2d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static constexpr uint32_t BLOCK_DIM = 16;

static inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }
static inline uint32_t CeilAlign(uint32_t x, uint32_t a) { return CeilDiv(x, a) * a; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MhcBlockBottleneck2dCustomTilingData t;

    const auto outShape = context->GetInputShape(0)->GetOriginShape(); // (B,C,H,W)
    const auto idShape  = context->GetInputShape(1)->GetOriginShape(); // (B,C,H,W)
    const auto mapShape = context->GetInputShape(2)->GetOriginShape(); // (B,S,S)

    if (outShape.GetDimNum() != 4 || idShape.GetDimNum() != 4 || mapShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t B = static_cast<uint32_t>(outShape.GetDim(0));
    const uint32_t C = static_cast<uint32_t>(outShape.GetDim(1));
    const uint32_t H = static_cast<uint32_t>(outShape.GetDim(2));
    const uint32_t W = static_cast<uint32_t>(outShape.GetDim(3));

    if ((uint32_t)idShape.GetDim(0) != B || (uint32_t)idShape.GetDim(1) != C ||
        (uint32_t)idShape.GetDim(2) != H || (uint32_t)idShape.GetDim(3) != W) {
        return ge::GRAPH_FAILED;
    }

    if ((uint32_t)mapShape.GetDim(0) != B) return ge::GRAPH_FAILED;
    const uint32_t S = static_cast<uint32_t>(mapShape.GetDim(1));
    if (S == 0 || S > 32) return ge::GRAPH_FAILED;
    if ((uint32_t)mapShape.GetDim(2) != S) return ge::GRAPH_FAILED;

    if (B == 0 || C == 0 || H == 0 || W == 0) return ge::GRAPH_FAILED;
    if (C % S != 0) return ge::GRAPH_FAILED;

    const uint32_t Cps = C / S;
    const uint32_t K = Cps * H * W;

    // UB conservative tile:
    // x(fp16 Kpad) + id(fp16 Kpad) + out(fp16 Kpad) + acc(fp32 Kpad) + a few small buffers.
    // Keep Kpad <= 4096 elements by default (8KB fp16, 16KB fp32 each).
    uint32_t tileK = K;
    const uint32_t maxTile = 4096;
    if (tileK > maxTile) tileK = maxTile;

    // Ensure >=16 and aligned for fp16 vector ops.
    tileK = CeilAlign(tileK, 16u);
    if (tileK == 0) tileK = 16;
    const uint32_t Kpad = tileK; // already aligned

    context->SetBlockDim(BLOCK_DIM);

    t.set_B(B); t.set_C(C); t.set_H(H); t.set_W(W);
    t.set_S(S); t.set_Cps(Cps); t.set_K(K);
    t.set_tileK(tileK); t.set_Kpad(Kpad);

    t.SaveToBuffer(context->GetRawTilingData()->GetData(),
                   context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(t.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x0 = context->GetInputShape(0);
    if (x0 == nullptr || x0->GetDimNum() != 4) return GRAPH_FAILED;
    gert::Shape* y = context->GetOutputShape(0);
    *y = *x0;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class MhcBlockBottleneck2dCustom : public OpDef {
public:
    explicit MhcBlockBottleneck2dCustom(const char* name) : OpDef(name)
    {
        this->Input("out_bn3").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("identity").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mapping_logits").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("sinkhorn_iter").ParamType(REQUIRED)
            .DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sinkhorn_eps").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sinkhorn_temperature").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MhcBlockBottleneck2dCustom);

} // namespace ops
