
#include "conv_transpose2d_bias_add_clamp_scaling_clamp_divide_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static inline int64_t ConvtOutDim(int64_t in, int64_t stride, int64_t pad, int64_t k, int64_t dil, int64_t out_pad) {
    return (in - 1) * stride - 2 * pad + dil * (k - 1) + out_pad + 1;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose2dBiasAddClampScalingClampDivideCustomTilingData t;

    const auto xShape  = context->GetInputShape(0)->GetStorageShape();
    const auto wShape  = context->GetInputShape(1)->GetStorageShape();
    const auto cbShape = context->GetInputShape(2)->GetStorageShape();
    const auto bShape  = context->GetInputShape(3)->GetStorageShape();

    if (xShape.GetDimNum() != 4 || wShape.GetDimNum() != 4) return ge::GRAPH_FAILED;
    if (cbShape.GetDimNum() != 1) return ge::GRAPH_FAILED;
    if (bShape.GetDimNum() != 3) return ge::GRAPH_FAILED;

    const uint32_t n   = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t cin = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t hin = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t win = static_cast<uint32_t>(xShape.GetDim(3));

    const uint32_t wcin = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t cout = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kh   = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t kw   = static_cast<uint32_t>(wShape.GetDim(3));

    if (wcin != cin) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(cbShape.GetDim(0)) != cout) return ge::GRAPH_FAILED;
    if (!(static_cast<uint32_t>(bShape.GetDim(0)) == cout &&
          static_cast<uint32_t>(bShape.GetDim(1)) == 1 &&
          static_cast<uint32_t>(bShape.GetDim(2)) == 1)) {
        return ge::GRAPH_FAILED;
    }

    // Specialization for this model
    constexpr uint32_t STR = 2;
    constexpr uint32_t PAD = 1;
    constexpr uint32_t OUT_PAD = 1;
    constexpr uint32_t DIL = 1;
    constexpr uint32_t K_EXPECT = 3;

    if (kh != K_EXPECT || kw != K_EXPECT) return ge::GRAPH_FAILED;
    if (!(n == 128 && cin == 64 && cout == 64 && hin == 128 && win == 128)) return ge::GRAPH_FAILED;

    const int64_t hout64 = ConvtOutDim(static_cast<int64_t>(hin), STR, PAD, static_cast<int64_t>(kh), DIL, OUT_PAD);
    const int64_t wout64 = ConvtOutDim(static_cast<int64_t>(win), STR, PAD, static_cast<int64_t>(kw), DIL, OUT_PAD);
    if (hout64 <= 0 || wout64 <= 0) return ge::GRAPH_FAILED;

    constexpr float SCALING = 2.0f;
    constexpr float CMIN = 0.0f;
    constexpr float CMAX = 1.0f;

    t.set_n(n);
    t.set_cin(cin);
    t.set_hin(hin);
    t.set_win(win);

    t.set_cout(cout);
    t.set_kh(kh);
    t.set_kw(kw);

    t.set_stride(STR);
    t.set_pad(PAD);
    t.set_out_pad(OUT_PAD);
    t.set_dilation(DIL);

    t.set_hout(static_cast<uint32_t>(hout64));
    t.set_wout(static_cast<uint32_t>(wout64));

    t.set_scaling(SCALING);
    t.set_clamp_min(CMIN);
    t.set_clamp_max(CMAX);

    // Stable mapping: one block per batch
    t.set_blocks(n);
    context->SetBlockDim(n);

    t.set_total_x(static_cast<uint32_t>(xShape.GetShapeSize()));
    t.set_total_w(static_cast<uint32_t>(wShape.GetShapeSize()));
    t.set_total_conv_bias(static_cast<uint32_t>(cbShape.GetShapeSize()));
    t.set_total_bias(static_cast<uint32_t>(bShape.GetShapeSize()));
    t.set_total_y(static_cast<uint32_t>(context->GetOutputShape(0)->GetStorageShape().GetShapeSize()));

    t.SaveToBuffer(context->GetRawTilingData()->GetData(),
                   context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(t.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class ConvTranspose2dBiasAddClampScalingClampDivideCustom : public OpDef {
public:
    explicit ConvTranspose2dBiasAddClampScalingClampDivideCustom(const char* name) : OpDef(name)
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

        this->Input("conv_bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose2dBiasAddClampScalingClampDivideCustom);

} // namespace ops
