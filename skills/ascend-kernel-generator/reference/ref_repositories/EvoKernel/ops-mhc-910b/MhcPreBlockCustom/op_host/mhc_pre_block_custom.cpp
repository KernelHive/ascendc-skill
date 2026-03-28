
#include "mhc_pre_block_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }
static inline float SafeInvU32(uint32_t v) { return (v == 0u) ? 1.0f : (1.0f / static_cast<float>(v)); }

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;

    const uint32_t totalResidual = static_cast<uint32_t>(context->GetInputShape(0)->GetOriginShape().GetShapeSize());
    const uint32_t totalFn       = static_cast<uint32_t>(context->GetInputShape(1)->GetOriginShape().GetShapeSize());
    const uint32_t totalScale    = static_cast<uint32_t>(context->GetInputShape(2)->GetOriginShape().GetShapeSize());
    const uint32_t totalBase     = static_cast<uint32_t>(context->GetInputShape(3)->GetOriginShape().GetShapeSize());

    uint32_t N = 1u, hc = 1u, H = 1u;
    auto resShape = context->GetInputShape(0)->GetOriginShape(); // (N,hc,H)
    auto fnShape  = context->GetInputShape(1)->GetOriginShape(); // (hc3, hc*H)

    bool ok = false;
    if (resShape.GetDimNum() == 3) {
        int64_t n0  = resShape.GetDim(0);
        int64_t hc0 = resShape.GetDim(1);
        int64_t h0  = resShape.GetDim(2);
        if (n0 > 0 && hc0 > 0 && h0 > 0) {
            N  = static_cast<uint32_t>(n0);
            hc = static_cast<uint32_t>(hc0);
            H  = static_cast<uint32_t>(h0);
            ok = true;
        }
    }

    if (!ok) {
        uint32_t hc3 = 0u, dFlat = 0u;
        if (fnShape.GetDimNum() == 2) {
            int64_t a = fnShape.GetDim(0);
            int64_t b = fnShape.GetDim(1);
            if (a > 0) hc3 = static_cast<uint32_t>(a);
            if (b > 0) dFlat = static_cast<uint32_t>(b);
        }
        if (hc3 == 0u && totalBase != 0u) hc3 = totalBase;

        uint32_t hc_guess = 1u;
        for (uint32_t t = 1u; t <= 128u; ++t) {
            if (t * t + 2u * t == hc3) { hc_guess = t; break; }
        }
        hc = hc_guess;

        if (dFlat == 0u && totalFn != 0u && hc3 != 0u) dFlat = totalFn / hc3;
        if (dFlat == 0u) dFlat = hc;

        H = (hc == 0u) ? 1u : (dFlat / hc);
        if (H == 0u) H = 1u;

        const uint32_t denom = hc * H;
        N = (denom == 0u) ? 1u : (totalResidual / denom);
        if (N == 0u) N = 1u;

        ok = true;
    }

    if (N == 0u) N = 1u;
    if (hc == 0u) hc = 1u;
    if (H == 0u) H = 1u;

    const uint32_t dFlat = hc * H;
    const uint32_t hc2 = hc * hc;
    const uint32_t hc3 = 2u * hc + hc2;

    // Keep many blocks; each token is compute-heavy but also bandwidth-heavy.
    uint32_t targetBlocks = 64u;
    uint32_t blockTokens = CeilDivU32(N, targetBlocks);
    if (blockTokens < 1u) blockTokens = 1u;
    if (blockTokens > 8u) blockTokens = 8u;

    uint32_t blockDim = CeilDivU32(N, blockTokens);
    if (blockDim < 1u) blockDim = 1u;
    if (blockDim > 4096u) blockDim = 4096u;
    context->SetBlockDim(blockDim);

    tiling.set_totalResidual(totalResidual);
    tiling.set_totalFn(totalFn);
    tiling.set_totalScale(totalScale);
    tiling.set_totalBase(totalBase);

    tiling.set_N(N);
    tiling.set_hc(hc);
    tiling.set_H(H);
    tiling.set_dFlat(dFlat);
    tiling.set_hc2(hc2);
    tiling.set_hc3(hc3);
    tiling.set_blockTokens(blockTokens);

    tiling.set_invDFlat(SafeInvU32(dFlat));
    tiling.set_rmsEps(1e-6f);
    tiling.set_hcPreEps(1e-6f);
    tiling.set_hcSinkhornEps(1e-6f);
    tiling.set_hcPostMultValue(1.0f);
    tiling.set_sinkhornRepeat(10u);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *res = context->GetInputShape(0); // (N,hc,H)
    int64_t N = 1, hc = 1, H = 1;
    if (res != nullptr && res->GetDimNum() == 3) {
        N = res->GetDim(0);
        hc = res->GetDim(1);
        H = res->GetDim(2);
    }

    gert::Shape *postShape = context->GetOutputShape(0);
    postShape->SetDimNum(3);
    postShape->SetDim(0, N);
    postShape->SetDim(1, hc);
    postShape->SetDim(2, 1);

    gert::Shape *combShape = context->GetOutputShape(1);
    combShape->SetDimNum(3);
    combShape->SetDim(0, N);
    combShape->SetDim(1, hc);
    combShape->SetDim(2, hc);

    gert::Shape *layerShape = context->GetOutputShape(2);
    layerShape->SetDimNum(2);
    layerShape->SetDim(0, N);
    layerShape->SetDim(1, H);

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class MhcPreBlockCustom : public OpDef {
public:
    explicit MhcPreBlockCustom(const char *name) : OpDef(name)
    {
        this->Input("residual").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("fn").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("hc_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("hc_base").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->Output("post_mix").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("comb_mix").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("layer_input").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(MhcPreBlockCustom);

} // namespace ops
