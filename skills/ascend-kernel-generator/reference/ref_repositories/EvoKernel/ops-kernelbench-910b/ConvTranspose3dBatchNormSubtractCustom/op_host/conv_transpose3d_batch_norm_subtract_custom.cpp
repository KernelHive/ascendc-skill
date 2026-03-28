
#include "conv_transpose3d_batch_norm_subtract_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static inline int64_t ConvtOutDim(int64_t in, int64_t stride, int64_t pad, int64_t k, int64_t dil, int64_t out_pad)
{
    return (in - 1) * stride - 2 * pad + dil * (k - 1) + out_pad + 1;
}

// Specialized benchmark contract
static constexpr uint32_t CIN  = 16;
static constexpr uint32_t DIN  = 16;
static constexpr uint32_t HIN  = 32;
static constexpr uint32_t WIN  = 32;

static constexpr uint32_t COUT = 32;
static constexpr uint32_t K    = 3;

static constexpr uint32_t STR  = 2;
static constexpr uint32_t PAD  = 1;
static constexpr uint32_t DIL  = 1;
static constexpr uint32_t OUTP = 0;

static constexpr float EPS = 1e-5f;
// Conservative blockDim to avoid over-parallelization regressions; keep channel-splitting.
static constexpr uint32_t MAX_BLOCK_DIM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose3dBatchNormSubtractCustomTilingData t;

    const auto* xShape = context->GetInputShape(0);
    if (xShape == nullptr) return ge::GRAPH_FAILED;
    const auto& xOs = xShape->GetOriginShape();
    if (xOs.GetDimNum() != 5) return ge::GRAPH_FAILED;

    const int64_t N64 = xOs.GetDim(0);
    const int64_t C64 = xOs.GetDim(1);
    const int64_t D64 = xOs.GetDim(2);
    const int64_t H64 = xOs.GetDim(3);
    const int64_t W64 = xOs.GetDim(4);
    if (N64 <= 0 || C64 <= 0 || D64 <= 0 || H64 <= 0 || W64 <= 0) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(C64) != CIN ||
        static_cast<uint32_t>(D64) != DIN ||
        static_cast<uint32_t>(H64) != HIN ||
        static_cast<uint32_t>(W64) != WIN) {
        return ge::GRAPH_FAILED;
    }

    // weight: PyTorch ConvTranspose3d layout [Cin, Cout, K, K, K]
    const auto* wShape = context->GetInputShape(1);
    if (wShape == nullptr) return ge::GRAPH_FAILED;
    const auto& wOs = wShape->GetOriginShape();
    if (wOs.GetDimNum() != 5) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(wOs.GetDim(0)) != CIN ||
        static_cast<uint32_t>(wOs.GetDim(1)) != COUT ||
        static_cast<uint32_t>(wOs.GetDim(2)) != K ||
        static_cast<uint32_t>(wOs.GetDim(3)) != K ||
        static_cast<uint32_t>(wOs.GetDim(4)) != K) {
        return ge::GRAPH_FAILED;
    }

    // conv_bias/bn_weight/bn_bias: [Cout]
    for (int idx = 2; idx <= 4; ++idx) {
        const auto* s = context->GetInputShape(static_cast<size_t>(idx));
        if (s == nullptr) return ge::GRAPH_FAILED;
        const auto& os = s->GetOriginShape();
        if (os.GetDimNum() != 1 || static_cast<uint32_t>(os.GetDim(0)) != COUT) return ge::GRAPH_FAILED;
    }

    const int64_t Dout64 = ConvtOutDim(DIN, STR, PAD, K, DIL, OUTP);
    const int64_t Hout64 = ConvtOutDim(HIN, STR, PAD, K, DIL, OUTP);
    const int64_t Wout64 = ConvtOutDim(WIN, STR, PAD, K, DIL, OUTP);

    if (Dout64 != 31 || Hout64 != 63 || Wout64 != 63) return ge::GRAPH_FAILED;

    uint32_t N = static_cast<uint32_t>(N64 > 0xFFFFFFFFLL ? 0xFFFFFFFFu : static_cast<uint32_t>(N64));
    if (N == 0) return ge::GRAPH_FAILED;

    const uint64_t DHW64 = static_cast<uint64_t>(Dout64) * static_cast<uint64_t>(Hout64) * static_cast<uint64_t>(Wout64);
    if (DHW64 == 0 || DHW64 > 0xFFFFFFFFULL) return ge::GRAPH_FAILED;
    const uint32_t DHW = static_cast<uint32_t>(DHW64);

    const uint64_t NHW64 = static_cast<uint64_t>(N) * DHW64;
    if (NHW64 == 0 || NHW64 > 0xFFFFFFFFULL) return ge::GRAPH_FAILED;
    const uint32_t NHW = static_cast<uint32_t>(NHW64);

    uint32_t blockDim = MAX_BLOCK_DIM;
    if (COUT > 0 && COUT < blockDim) blockDim = COUT;
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    t.set_Cin(CIN);
    t.set_Din(DIN);
    t.set_Hin(HIN);
    t.set_Win(WIN);

    t.set_Cout(COUT);
    t.set_K(K);

    t.set_Stride(STR);
    t.set_Pad(PAD);
    t.set_Dil(DIL);
    t.set_OutPad(OUTP);

    t.set_Dout(static_cast<uint32_t>(Dout64));
    t.set_Hout(static_cast<uint32_t>(Hout64));
    t.set_Wout(static_cast<uint32_t>(Wout64));

    t.set_DHW(DHW);
    t.set_NHW(NHW);
    t.set_invDHW(1.0f / static_cast<float>(DHW));
    t.set_invNHW(1.0f / static_cast<float>(NHW));
    t.set_eps(EPS);
    t.set_N(N);

    t.SaveToBuffer(context->GetRawTilingData()->GetData(),
                   context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(t.GetDataSize());

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x = context->GetInputShape(0);
    const gert::Shape* w = context->GetInputShape(1);
    if (x == nullptr || w == nullptr) return GRAPH_FAILED;
    if (x->GetDimNum() != 5 || w->GetDimNum() != 5) return GRAPH_FAILED;

    gert::Shape* y = context->GetOutputShape(0);
    if (y == nullptr) return GRAPH_FAILED;

    y->SetDimNum(5);
    y->SetDim(0, x->GetDim(0));
    y->SetDim(1, w->GetDim(1)); // Cout
    y->SetDim(2, 31);
    y->SetDim(3, 63);
    y->SetDim(4, 63);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class ConvTranspose3dBatchNormSubtractCustom : public OpDef {
public:
    explicit ConvTranspose3dBatchNormSubtractCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("conv_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bn_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bn_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose3dBatchNormSubtractCustom);

}  // namespace ops
