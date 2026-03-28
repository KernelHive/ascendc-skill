
#include "conv3d_group_norm_min_clamp_dropout_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <cmath>

namespace optiling {

// Specialized benchmark contract (fixed except N)
static constexpr uint32_t CIN  = 3;
static constexpr uint32_t DIN  = 16;
static constexpr uint32_t HIN  = 64;
static constexpr uint32_t WIN  = 64;

static constexpr uint32_t COUT = 16;
static constexpr uint32_t K    = 3;

// Conv3d: stride=1, pad=0, dilation=1
static constexpr uint32_t STR = 1;
static constexpr uint32_t PAD = 0;
static constexpr uint32_t DIL = 1;

static constexpr uint32_t DOUT = (DIN + 2 * PAD - DIL * (K - 1) - 1) / STR + 1; // 14
static constexpr uint32_t HOUT = (HIN + 2 * PAD - DIL * (K - 1) - 1) / STR + 1; // 62
static constexpr uint32_t WOUT = (WIN + 2 * PAD - DIL * (K - 1) - 1) / STR + 1; // 62

static constexpr uint32_t G = 8;
static constexpr float EPS = 1e-5f;

// Min and clamp are specialized exactly to min(.,0) then clamp [0,1]
static constexpr float MIN_VALUE = 0.0f;
static constexpr float CLAMP_MIN = 0.0f;
static constexpr float CLAMP_MAX = 1.0f;

// Dropout always applied (training-like), p=0.2
static constexpr float DROPOUT_P = 0.2f;

// Raise occupancy: map blocks over N*G tasks
static constexpr uint32_t MAX_BLOCK_DIM = 96;

static inline uint32_t DropThresholdU32(float p)
{
    if (p <= 0.0f) return 0u;
    if (p >= 1.0f) return 0xFFFFFFFFu;
    const double scale = 4294967296.0; // 2^32
    double v = static_cast<double>(p) * scale;
    if (v < 0.0) v = 0.0;
    if (v > 4294967295.0) v = 4294967295.0;
    return static_cast<uint32_t>(v);
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv3dGroupNormMinClampDropoutCustomTilingData tiling;

    const auto* xShape = context->GetInputShape(0);
    const auto* wShape = context->GetInputShape(1);
    const auto* bShape = context->GetInputShape(2);
    const auto* gShape = context->GetInputShape(3);
    const auto* beShape = context->GetInputShape(4);
    if (!xShape || !wShape || !bShape || !gShape || !beShape) return ge::GRAPH_FAILED;

    const auto& xOs = xShape->GetOriginShape();
    const auto& wOs = wShape->GetOriginShape();
    const auto& bOs = bShape->GetOriginShape();
    const auto& gOs = gShape->GetOriginShape();
    const auto& beOs = beShape->GetOriginShape();

    if (xOs.GetDimNum() != 5 || wOs.GetDimNum() != 5) return ge::GRAPH_FAILED;
    if (bOs.GetDimNum() != 1 || gOs.GetDimNum() != 1 || beOs.GetDimNum() != 1) return ge::GRAPH_FAILED;

    const int64_t N64 = xOs.GetDim(0);
    if (N64 <= 0) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(xOs.GetDim(1)) != CIN ||
        static_cast<uint32_t>(xOs.GetDim(2)) != DIN ||
        static_cast<uint32_t>(xOs.GetDim(3)) != HIN ||
        static_cast<uint32_t>(xOs.GetDim(4)) != WIN) {
        return ge::GRAPH_FAILED;
    }

    // weight: [Cout,Cin,K,K,K]
    if (static_cast<uint32_t>(wOs.GetDim(0)) != COUT ||
        static_cast<uint32_t>(wOs.GetDim(1)) != CIN ||
        static_cast<uint32_t>(wOs.GetDim(2)) != K ||
        static_cast<uint32_t>(wOs.GetDim(3)) != K ||
        static_cast<uint32_t>(wOs.GetDim(4)) != K) {
        return ge::GRAPH_FAILED;
    }

    if (static_cast<uint32_t>(bOs.GetDim(0)) != COUT) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(gOs.GetDim(0)) != COUT) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(beOs.GetDim(0)) != COUT) return ge::GRAPH_FAILED;

    if (G == 0 || (COUT % G) != 0) return ge::GRAPH_FAILED;
    const uint32_t CperG = COUT / G;

    const uint32_t DHW = DOUT * HOUT * WOUT;
    const uint32_t elemsPerG = CperG * DHW;
    if (DHW == 0 || elemsPerG == 0) return ge::GRAPH_FAILED;

    uint32_t N = static_cast<uint32_t>(N64 > 0xFFFFFFFFLL ? 0xFFFFFFFFu : static_cast<uint32_t>(N64));

    // tasks = N*G; cap blockDim conservatively
    uint64_t tasks64 = static_cast<uint64_t>(N) * static_cast<uint64_t>(G);
    uint32_t tasks = static_cast<uint32_t>(tasks64 > 0xFFFFFFFFULL ? 0xFFFFFFFFu : static_cast<uint32_t>(tasks64));

    uint32_t blockDim = MAX_BLOCK_DIM;
    if (tasks > 0 && tasks < blockDim) blockDim = tasks;
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    const float keepProb = 1.0f - DROPOUT_P;
    if (!(keepProb > 0.0f)) return ge::GRAPH_FAILED;
    const float invKeepProb = 1.0f / keepProb;

    tiling.set_N(N);

    tiling.set_Cin(CIN);
    tiling.set_Din(DIN);
    tiling.set_Hin(HIN);
    tiling.set_Win(WIN);

    tiling.set_Cout(COUT);
    tiling.set_K(K);

    tiling.set_Dout(DOUT);
    tiling.set_Hout(HOUT);
    tiling.set_Wout(WOUT);

    tiling.set_G(G);
    tiling.set_CperG(CperG);

    tiling.set_DHW(DHW);
    tiling.set_elemsPerG(elemsPerG);
    tiling.set_invElemsPerG(1.0f / static_cast<float>(elemsPerG));
    tiling.set_eps(EPS);

    tiling.set_minValue(MIN_VALUE);
    tiling.set_clampMin(CLAMP_MIN);
    tiling.set_clampMax(CLAMP_MAX);

    tiling.set_dropoutP(DROPOUT_P);
    tiling.set_dropThresholdU32(DropThresholdU32(DROPOUT_P));
    tiling.set_invKeepProb(invKeepProb);

    tiling.set_tasksPerSample(G);

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
    const gert::Shape* x = context->GetInputShape(0);
    if (x == nullptr) return GRAPH_FAILED;
    if (x->GetDimNum() != 5) return GRAPH_FAILED;

    gert::Shape* y = context->GetOutputShape(0);
    if (y == nullptr) return GRAPH_FAILED;

    // specialized output: [N,16,14,62,62]
    y->SetDimNum(5);
    y->SetDim(0, x->GetDim(0));
    y->SetDim(1, 16);
    y->SetDim(2, 14);
    y->SetDim(3, 62);
    y->SetDim(4, 62);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class Conv3dGroupNormMinClampDropoutCustom : public OpDef {
public:
    explicit Conv3dGroupNormMinClampDropoutCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("gn_gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("gn_beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv3dGroupNormMinClampDropoutCustom);

} // namespace ops
