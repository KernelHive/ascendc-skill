
#include "fused_mhc_kernels_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cmath>
#include <cstdint>

namespace optiling {

static inline uint32_t Ceil8(uint32_t x) { return ((x + 7u) / 8u) * 8u; }

static inline uint32_t InferN(uint32_t outCols)
{
    // outCols = n*n + 2n => n = floor(sqrt(outCols + 1) - 1)
    const double v = std::sqrt(static_cast<double>(outCols) + 1.0);
    if (v < 2.0) return 0U;
    const uint32_t n = static_cast<uint32_t>(std::floor(v + 1e-9)) - 1U;
    return n;
}

// Fused op is compute-heavy per row; moderate block dim helps amortize weight streaming.
static constexpr uint32_t BLOCK_DIM = 16;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    FusedMhcKernelsCustomTilingData tiling;

    const auto xShape   = context->GetInputShape(0)->GetOriginShape(); // (B,L,D)
    const auto pShape   = context->GetInputShape(1)->GetOriginShape(); // (D,outCols)
    const auto bShape   = context->GetInputShape(2)->GetOriginShape(); // (outCols) or (1,outCols)
    const auto sShape   = context->GetInputShape(3)->GetOriginShape(); // (D)
    const auto apShape  = context->GetInputShape(4)->GetOriginShape(); // (1)
    const auto aoShape  = context->GetInputShape(5)->GetOriginShape(); // (1)
    const auto arShape  = context->GetInputShape(6)->GetOriginShape(); // (1)
    const auto itShape  = context->GetInputShape(7)->GetOriginShape(); // (1)
    const auto erShape  = context->GetInputShape(8)->GetOriginShape(); // (1)
    const auto invShape = context->GetInputShape(9)->GetOriginShape(); // (1)

    if (xShape.GetDimNum() != 3 || pShape.GetDimNum() != 2 || sShape.GetDimNum() != 1) return ge::GRAPH_FAILED;
    if (bShape.GetDimNum() != 1 && bShape.GetDimNum() != 2) return ge::GRAPH_FAILED;
    if (apShape.GetDimNum() < 1 || aoShape.GetDimNum() < 1 || arShape.GetDimNum() < 1 ||
        itShape.GetDimNum() < 1 || erShape.GetDimNum() < 1 || invShape.GetDimNum() < 1) return ge::GRAPH_FAILED;

    const uint32_t B = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t L = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t D = static_cast<uint32_t>(xShape.GetDim(2));
    if (B == 0 || L == 0 || D == 0) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(pShape.GetDim(0)) != D) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(sShape.GetDim(0)) != D) return ge::GRAPH_FAILED;

    const uint32_t outCols = static_cast<uint32_t>(pShape.GetDim(1));
    if (outCols == 0) return ge::GRAPH_FAILED;

    const uint32_t n = InferN(outCols);
    if (n == 0U) return ge::GRAPH_FAILED;
    if ((n * n + 2U * n) != outCols) return ge::GRAPH_FAILED;
    const uint32_t nn = n * n;

    // bias must have outCols elements (accept (outCols) or (1,outCols))
    uint64_t biasElems = 1;
    for (uint32_t i = 0; i < static_cast<uint32_t>(bShape.GetDimNum()); ++i) {
        biasElems *= static_cast<uint64_t>(bShape.GetDim(i));
    }
    if (biasElems != static_cast<uint64_t>(outCols)) return ge::GRAPH_FAILED;

    context->SetBlockDim(BLOCK_DIM);

    const uint32_t BL = B * L;
    const uint32_t Dpad   = Ceil8(D);
    const uint32_t outPad = Ceil8(outCols);
    const uint32_t nPad   = Ceil8(n);
    const uint32_t nnPad  = Ceil8(nn);

    tiling.set_B(B);
    tiling.set_L(L);
    tiling.set_D(D);
    tiling.set_BL(BL);
    tiling.set_outCols(outCols);
    tiling.set_n(n);
    tiling.set_nn(nn);
    tiling.set_Dpad(Dpad);
    tiling.set_outPad(outPad);
    tiling.set_nPad(nPad);
    tiling.set_nnPad(nnPad);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static inline int64_t InferNFromOutCols(int64_t outCols)
{
    const double v = std::sqrt(static_cast<double>(outCols) + 1.0);
    if (v < 2.0) return 0;
    return static_cast<int64_t>(std::floor(v + 1e-9)) - 1;
}

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x = context->GetInputShape(0);
    const gert::Shape* p = context->GetInputShape(1);
    if (x == nullptr || p == nullptr) return GRAPH_FAILED;
    if (x->GetDimNum() != 3 || p->GetDimNum() != 2) return GRAPH_FAILED;

    const int64_t B = x->GetDim(0);
    const int64_t L = x->GetDim(1);
    const int64_t outCols = p->GetDim(1);

    const int64_t n = InferNFromOutCols(outCols);
    if (n <= 0 || (n * n + 2 * n) != outCols) return GRAPH_FAILED;

    gert::Shape* o0 = context->GetOutputShape(0);
    gert::Shape* o1 = context->GetOutputShape(1);
    gert::Shape* o2 = context->GetOutputShape(2);

    o0->SetDimNum(3); o0->SetDim(0, B); o0->SetDim(1, L); o0->SetDim(2, n);
    o1->SetDimNum(3); o1->SetDim(0, B); o1->SetDim(1, L); o1->SetDim(2, n);
    o2->SetDimNum(4); o2->SetDim(0, B); o2->SetDim(1, L); o2->SetDim(2, n); o2->SetDim(3, n);
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

class FusedMhcKernelsCustom : public OpDef {
public:
    explicit FusedMhcKernelsCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("phi").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("alpha_pre").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("alpha_post").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("alpha_res").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("iters").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("eps_rms").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("invD").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("h_pre").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("h_post").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("h_res").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(FusedMhcKernelsCustom);

} // namespace ops
