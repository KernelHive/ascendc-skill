
#include "optimized_mhc_layer_with_fusion_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (b == 0u) ? 0u : ((a + b - 1u) / b); }
static inline float SafeInvU32(uint32_t v) { return (v == 0u) ? 1.0f : (1.0f / static_cast<float>(v)); }

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData t;

    const uint32_t totalX         = static_cast<uint32_t>(context->GetInputShape(0)->GetOriginShape().GetShapeSize());
    const uint32_t totalPhi       = static_cast<uint32_t>(context->GetInputShape(1)->GetOriginShape().GetShapeSize());
    const uint32_t totalBias      = static_cast<uint32_t>(context->GetInputShape(2)->GetOriginShape().GetShapeSize());
    const uint32_t totalRmsScale  = static_cast<uint32_t>(context->GetInputShape(3)->GetOriginShape().GetShapeSize());
    const uint32_t totalAlphaPre  = static_cast<uint32_t>(context->GetInputShape(4)->GetOriginShape().GetShapeSize());
    const uint32_t totalAlphaPost = static_cast<uint32_t>(context->GetInputShape(5)->GetOriginShape().GetShapeSize());
    const uint32_t totalAlphaRes  = static_cast<uint32_t>(context->GetInputShape(6)->GetOriginShape().GetShapeSize());
    const uint32_t totalW         = static_cast<uint32_t>(context->GetInputShape(7)->GetOriginShape().GetShapeSize());

    // x: (B,S,D), but we infer SD from phi/rms if needed and specialize to n=4
    uint32_t B = 1u, S = 1u, D = 1u;
    auto xShape = context->GetInputShape(0)->GetOriginShape();
    if (xShape.GetDimNum() == 3) {
        int64_t b = xShape.GetDim(0);
        int64_t s = xShape.GetDim(1);
        int64_t d = xShape.GetDim(2);
        if (b > 0) B = static_cast<uint32_t>(b);
        if (s > 0) S = static_cast<uint32_t>(s);
        if (d > 0) D = static_cast<uint32_t>(d);
    }

    // Determine n and SD using phi (SD,mapDim) and rms_scale (SD)
    uint32_t SD = 0u;
    uint32_t mapDim = 0u;
    auto phiShape = context->GetInputShape(1)->GetOriginShape();
    if (phiShape.GetDimNum() == 2) {
        int64_t sd0 = phiShape.GetDim(0);
        int64_t md0 = phiShape.GetDim(1);
        if (sd0 > 0) SD = static_cast<uint32_t>(sd0);
        if (md0 > 0) mapDim = static_cast<uint32_t>(md0);
    }
    if (mapDim == 0u && totalBias != 0u) mapDim = totalBias;

    // Infer n from mapDim = n*n + 2*n
    uint32_t n = 4u;
    if (mapDim != 0u) {
        for (uint32_t t0 = 1u; t0 <= 64u; ++t0) {
            if (t0 * t0 + 2u * t0 == mapDim) { n = t0; break; }
        }
    }

    // If SD is missing, infer from rms_scale length
    if (SD == 0u && totalRmsScale != 0u) SD = totalRmsScale;

    // If still missing, derive SD from n*D
    if (SD == 0u) SD = n * D;

    // Infer D from SD/n if x provided D may differ; keep D from x if non-zero.
    if (D == 0u) D = (n == 0u) ? 1u : (SD / n);
    if (D == 0u) D = 1u;

    const uint32_t tokens = B * S;

    uint32_t targetBlocks = 256u;
    uint32_t tokensPerCore = CeilDivU32(tokens, targetBlocks);
    if (tokensPerCore < 1u) tokensPerCore = 1u;
    if (tokensPerCore > 2u) tokensPerCore = 2u;

    uint32_t blockDim = CeilDivU32(tokens, tokensPerCore);
    if (blockDim < 1u) blockDim = 1u;
    if (blockDim > 4096u) blockDim = 4096u;
    context->SetBlockDim(blockDim);

    t.set_totalX(totalX);
    t.set_totalPhi(totalPhi);
    t.set_totalBias(totalBias);
    t.set_totalRmsScale(totalRmsScale);
    t.set_totalAlphaPre(totalAlphaPre);
    t.set_totalAlphaPost(totalAlphaPost);
    t.set_totalAlphaRes(totalAlphaRes);
    t.set_totalW(totalW);

    t.set_B(B);
    t.set_S(S);
    t.set_D(D);
    t.set_n(n);
    t.set_SD(SD);
    t.set_mapDim(mapDim);
    t.set_tokens(tokens);
    t.set_tokensPerCore(tokensPerCore);

    t.set_invSD(SafeInvU32(SD));
    t.set_rmsEps(1e-20f);
    t.set_sinkEps(1e-12f);
    t.set_sinkIters(20u);

    t.SaveToBuffer(context->GetRawTilingData()->GetData(),
                   context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(t.GetDataSize());

    size_t *ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    // Output y: (B,S,D)
    const gert::Shape *x = context->GetInputShape(0);
    int64_t B = 1, S = 1, D = 1;
    if (x != nullptr && x->GetDimNum() == 3) {
        B = x->GetDim(0);
        S = x->GetDim(1);
        D = x->GetDim(2);
    }
    gert::Shape *y = context->GetOutputShape(0);
    y->SetDimNum(3);
    y->SetDim(0, B);
    y->SetDim(1, S);
    y->SetDim(2, D);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class OptimizedMHCLayerWithFusionCustom : public OpDef {
public:
    explicit OptimizedMHCLayerWithFusionCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("phi_params").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias_params").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("rms_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("alpha_pre").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("alpha_post").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("alpha_res").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("linear_w").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(OptimizedMHCLayerWithFusionCustom);

} // namespace ops
