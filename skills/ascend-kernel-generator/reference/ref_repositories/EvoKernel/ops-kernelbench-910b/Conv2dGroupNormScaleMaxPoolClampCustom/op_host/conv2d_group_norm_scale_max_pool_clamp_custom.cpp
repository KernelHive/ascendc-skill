
#include "conv2d_group_norm_scale_max_pool_clamp_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Specialized benchmark constants
static constexpr uint32_t CIN  = 8;
static constexpr uint32_t HIN  = 128;
static constexpr uint32_t WIN  = 128;

static constexpr uint32_t COUT = 64;
static constexpr uint32_t K    = 3;

static constexpr uint32_t STR  = 1;
static constexpr uint32_t PAD  = 0;
static constexpr uint32_t DIL  = 1;

static constexpr uint32_t HC = (HIN + 2 * PAD - DIL * (K - 1) - 1) / STR + 1; // 126
static constexpr uint32_t WC = (WIN + 2 * PAD - DIL * (K - 1) - 1) / STR + 1; // 126

static constexpr uint32_t POOL_K = 4;
static constexpr uint32_t POOL_S = 4;
static constexpr uint32_t HO = (HC - POOL_K) / POOL_S + 1; // 31
static constexpr uint32_t WO = (WC - POOL_K) / POOL_S + 1; // 31

static constexpr uint32_t G = 16;
static constexpr float EPS = 1e-5f;
static constexpr float CLAMP_MIN = 0.0f;
static constexpr float CLAMP_MAX = 1.0f;

// increased occupancy vs previous (still per-N mapping, independent blocks)
static constexpr uint32_t MAX_BLOCK_DIM = 64;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dGroupNormScaleMaxPoolClampCustomTilingData tiling;

    const auto* xShape = context->GetInputShape(0);
    if (xShape == nullptr) return ge::GRAPH_FAILED;
    const auto& xOs = xShape->GetOriginShape();
    if (xOs.GetDimNum() != 4) return ge::GRAPH_FAILED;

    const int64_t N64  = xOs.GetDim(0);
    const int64_t C64  = xOs.GetDim(1);
    const int64_t H64  = xOs.GetDim(2);
    const int64_t W64  = xOs.GetDim(3);
    if (N64 <= 0 || C64 <= 0 || H64 <= 0 || W64 <= 0) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(C64) != CIN ||
        static_cast<uint32_t>(H64) != HIN ||
        static_cast<uint32_t>(W64) != WIN) {
        return ge::GRAPH_FAILED;
    }

    const auto* wShape = context->GetInputShape(1);
    if (wShape == nullptr) return ge::GRAPH_FAILED;
    const auto& wOs = wShape->GetOriginShape();
    if (wOs.GetDimNum() != 4) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(wOs.GetDim(0)) != COUT ||
        static_cast<uint32_t>(wOs.GetDim(1)) != CIN ||
        static_cast<uint32_t>(wOs.GetDim(2)) != K ||
        static_cast<uint32_t>(wOs.GetDim(3)) != K) {
        return ge::GRAPH_FAILED;
    }

    auto check1d = [&](int idx, uint32_t expected) -> bool {
        const auto* s = context->GetInputShape(idx);
        if (s == nullptr) return false;
        const auto& os = s->GetOriginShape();
        if (os.GetDimNum() != 1) return false;
        return static_cast<uint32_t>(os.GetDim(0)) == expected;
    };

    if (!check1d(2, COUT)) return ge::GRAPH_FAILED; // bias
    if (!check1d(3, COUT)) return ge::GRAPH_FAILED; // gn_gamma
    if (!check1d(4, COUT)) return ge::GRAPH_FAILED; // gn_beta
    if (!check1d(5, COUT)) return ge::GRAPH_FAILED; // scale

    uint32_t N = static_cast<uint32_t>(N64 > 0xFFFFFFFFLL ? 0xFFFFFFFFu : static_cast<uint32_t>(N64));

    if (G == 0 || (COUT % G) != 0) return ge::GRAPH_FAILED;
    const uint32_t CperG = COUT / G;

    const uint32_t elemsPerG = CperG * HC * WC; // 4*126*126 = 63504
    if (elemsPerG == 0) return ge::GRAPH_FAILED;

    uint32_t blockDim = MAX_BLOCK_DIM;
    if (N > 0 && N < blockDim) blockDim = N;
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    tiling.set_Cin(CIN);
    tiling.set_Hin(HIN);
    tiling.set_Win(WIN);

    tiling.set_Cout(COUT);
    tiling.set_K(K);

    tiling.set_Hc(HC);
    tiling.set_Wc(WC);

    tiling.set_Ho(HO);
    tiling.set_Wo(WO);

    tiling.set_G(G);
    tiling.set_CperG(CperG);

    tiling.set_poolK(POOL_K);
    tiling.set_poolS(POOL_S);

    tiling.set_elemsPerG(elemsPerG);
    tiling.set_invElemsPerG(1.0f / static_cast<float>(elemsPerG));
    tiling.set_eps(EPS);

    tiling.set_clampMin(CLAMP_MIN);
    tiling.set_clampMax(CLAMP_MAX);

    tiling.set_N(N);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x = context->GetInputShape(0);
    if (x == nullptr) return GRAPH_FAILED;
    if (x->GetDimNum() != 4) return GRAPH_FAILED;

    gert::Shape* y = context->GetOutputShape(0);
    if (y == nullptr) return GRAPH_FAILED;

    // specialized output: [N,64,31,31]
    y->SetDimNum(4);
    y->SetDim(0, x->GetDim(0));
    y->SetDim(1, 64);
    y->SetDim(2, 31);
    y->SetDim(3, 31);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Conv2dGroupNormScaleMaxPoolClampCustom : public OpDef {
public:
    explicit Conv2dGroupNormScaleMaxPoolClampCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gn_gamma").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gn_beta").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dGroupNormScaleMaxPoolClampCustom);

}  // namespace ops
