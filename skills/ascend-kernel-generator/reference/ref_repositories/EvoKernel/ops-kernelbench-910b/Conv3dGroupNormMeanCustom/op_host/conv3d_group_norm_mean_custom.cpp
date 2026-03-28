
#include "conv3d_group_norm_mean_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Baked specialization for benchmark
static constexpr uint32_t CIN  = 3;
static constexpr uint32_t DIN  = 24;
static constexpr uint32_t HIN  = 32;
static constexpr uint32_t WIN  = 32;

static constexpr uint32_t COUT = 24;
static constexpr uint32_t K    = 3;
static constexpr uint32_t G    = 8;

static constexpr uint32_t DOUT = DIN - K + 1; // 22
static constexpr uint32_t HOUT = HIN - K + 1; // 30
static constexpr uint32_t WOUT = WIN - K + 1; // 30

static constexpr float EPS = 1e-5f;

// Conservative launch to avoid device bring-up issues
static constexpr uint32_t MAX_BLOCK_DIM = 8;

// Still provide a small UB scratch size (ABI stable)
static constexpr uint32_t TILE_DHW_DEFAULT = 256;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv3dGroupNormMeanCustomTilingData tiling;

    // x: [N,C,D,H,W]
    const auto* xShape = context->GetInputShape(0);
    const auto& xOs = xShape->GetOriginShape();
    if (xOs.GetDimNum() != 5) return ge::GRAPH_FAILED;

    const int64_t N64 = xOs.GetDim(0);
    const int64_t C64 = xOs.GetDim(1);
    const int64_t D64 = xOs.GetDim(2);
    const int64_t H64 = xOs.GetDim(3);
    const int64_t W64 = xOs.GetDim(4);

    if (N64 <= 0 || C64 <= 0 || D64 <= 0 || H64 <= 0 || W64 <= 0) return ge::GRAPH_FAILED;

    // Enforce specialization
    if (static_cast<uint32_t>(C64) != CIN ||
        static_cast<uint32_t>(D64) != DIN ||
        static_cast<uint32_t>(H64) != HIN ||
        static_cast<uint32_t>(W64) != WIN) {
        return ge::GRAPH_FAILED;
    }

    // weight: [Cout,Cin,K,K,K]
    const auto* wShape = context->GetInputShape(1);
    const auto& wOs = wShape->GetOriginShape();
    if (wOs.GetDimNum() != 5) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(wOs.GetDim(0)) != COUT ||
        static_cast<uint32_t>(wOs.GetDim(1)) != CIN  ||
        static_cast<uint32_t>(wOs.GetDim(2)) != K    ||
        static_cast<uint32_t>(wOs.GetDim(3)) != K    ||
        static_cast<uint32_t>(wOs.GetDim(4)) != K) {
        return ge::GRAPH_FAILED;
    }

    // bias/gamma/beta: [Cout]
    const auto* bShape = context->GetInputShape(2);
    const auto& bOs = bShape->GetOriginShape();
    if (bOs.GetDimNum() != 1 || static_cast<uint32_t>(bOs.GetDim(0)) != COUT) return ge::GRAPH_FAILED;

    const auto* gShape = context->GetInputShape(3);
    const auto& gOs = gShape->GetOriginShape();
    if (gOs.GetDimNum() != 1 || static_cast<uint32_t>(gOs.GetDim(0)) != COUT) return ge::GRAPH_FAILED;

    const auto* beShape = context->GetInputShape(4);
    const auto& beOs = beShape->GetOriginShape();
    if (beOs.GetDimNum() != 1 || static_cast<uint32_t>(beOs.GetDim(0)) != COUT) return ge::GRAPH_FAILED;

    uint32_t N = static_cast<uint32_t>(N64 > 0xFFFFFFFFLL ? 0xFFFFFFFFu : static_cast<uint32_t>(N64));

    if (G == 0 || (COUT % G) != 0) return ge::GRAPH_FAILED;
    const uint32_t CperG = COUT / G;

    const uint32_t DHW = DOUT * HOUT * WOUT;
    const uint32_t elemsPerG = CperG * DHW;
    const uint32_t elemsPerN = COUT * DHW;
    if (DHW == 0 || elemsPerG == 0 || elemsPerN == 0) return ge::GRAPH_FAILED;

    uint32_t tileDhw = TILE_DHW_DEFAULT;
    if (tileDhw > DHW) tileDhw = DHW;
    if (tileDhw == 0) tileDhw = 1;

    uint32_t blockDim = MAX_BLOCK_DIM;
    if (N > 0 && N < blockDim) blockDim = N;
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

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
    tiling.set_elemsPerN(elemsPerN);

    tiling.set_tileDhw(tileDhw);

    tiling.set_invElemsPerG(1.0f / static_cast<float>(elemsPerG));
    tiling.set_invElemsPerN(1.0f / static_cast<float>(elemsPerN));
    tiling.set_eps(EPS);

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
    // Output is [N]
    const gert::Shape* x = context->GetInputShape(0);
    if (x == nullptr) return GRAPH_FAILED;
    if (x->GetDimNum() != 5) return GRAPH_FAILED;

    gert::Shape* y = context->GetOutputShape(0);
    if (y == nullptr) return GRAPH_FAILED;

    y->SetDimNum(1);
    y->SetDim(0, x->GetDim(0));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Conv3dGroupNormMeanCustom : public OpDef {
public:
    explicit Conv3dGroupNormMeanCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("beta").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv3dGroupNormMeanCustom);

}  // namespace ops
