
#include "gru_bidirectional_hidden_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GruBidirectionalHiddenCustomTilingData tiling;

    const auto xShape   = context->GetInputShape(0)->GetStorageShape();
    const auto h0Shape  = context->GetInputShape(1)->GetStorageShape();
    const auto wihShape = context->GetInputShape(2)->GetStorageShape();
    const auto whhShape = context->GetInputShape(3)->GetStorageShape();
    const auto bihShape = context->GetInputShape(4)->GetStorageShape();
    const auto bhhShape = context->GetInputShape(5)->GetStorageShape();
    const auto ybShape  = context->GetInputShape(6)->GetStorageShape();
    const auto outShape = context->GetOutputShape(0)->GetStorageShape();

    if (xShape.GetDimNum() != 3 || h0Shape.GetDimNum() != 3 ||
        wihShape.GetDimNum() != 2 || whhShape.GetDimNum() != 2 ||
        bihShape.GetDimNum() != 1 || bhhShape.GetDimNum() != 1 ||
        ybShape.GetDimNum() != 4 || outShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    constexpr uint32_t T_EXP = 512;
    constexpr uint32_t B_EXP = 10;
    constexpr uint32_t I_EXP = 128;
    constexpr uint32_t H_EXP = 256;
    constexpr uint32_t L_EXP = 6;
    constexpr uint32_t D_EXP = 2;
    constexpr uint32_t IN2H  = 2U * H_EXP; // 512
    constexpr uint32_t ROWS  = L_EXP * D_EXP * 3U * H_EXP; // 9216

    const uint32_t T = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t B = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t I = static_cast<uint32_t>(xShape.GetDim(2));
    if (T != T_EXP || B != B_EXP || I != I_EXP) return ge::GRAPH_FAILED;

    const uint32_t LD = static_cast<uint32_t>(h0Shape.GetDim(0));
    const uint32_t B2 = static_cast<uint32_t>(h0Shape.GetDim(1));
    const uint32_t H  = static_cast<uint32_t>(h0Shape.GetDim(2));
    if (LD != L_EXP * D_EXP || B2 != B_EXP || H != H_EXP) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(wihShape.GetDim(0)) != ROWS ||
        static_cast<uint32_t>(wihShape.GetDim(1)) != IN2H) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(whhShape.GetDim(0)) != ROWS ||
        static_cast<uint32_t>(whhShape.GetDim(1)) != H_EXP) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(bihShape.GetDim(0)) != ROWS ||
        static_cast<uint32_t>(bhhShape.GetDim(0)) != ROWS) return ge::GRAPH_FAILED;

    // ybuf is kept as an input for API compatibility; kernel may ignore it.
    if (static_cast<uint32_t>(ybShape.GetDim(0)) != L_EXP ||
        static_cast<uint32_t>(ybShape.GetDim(1)) != T_EXP ||
        static_cast<uint32_t>(ybShape.GetDim(2)) != B_EXP ||
        static_cast<uint32_t>(ybShape.GetDim(3)) != IN2H) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(outShape.GetDim(0)) != L_EXP * D_EXP ||
        static_cast<uint32_t>(outShape.GetDim(1)) != B_EXP ||
        static_cast<uint32_t>(outShape.GetDim(2)) != H_EXP) return ge::GRAPH_FAILED;

    tiling.set_T(T_EXP);
    tiling.set_B(B_EXP);
    tiling.set_I(I_EXP);
    tiling.set_H(H_EXP);
    tiling.set_L(L_EXP);
    tiling.set_D(D_EXP);

    // One block per batch element; each block processes full sequence and all layers for its batch slice.
    context->SetBlockDim(B_EXP);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class GruBidirectionalHiddenCustom : public OpDef {
public:
    explicit GruBidirectionalHiddenCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("h0").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w_ih").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w_hh").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_ih").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_hh").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ybuf").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("h_n").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GruBidirectionalHiddenCustom);

} // namespace ops
