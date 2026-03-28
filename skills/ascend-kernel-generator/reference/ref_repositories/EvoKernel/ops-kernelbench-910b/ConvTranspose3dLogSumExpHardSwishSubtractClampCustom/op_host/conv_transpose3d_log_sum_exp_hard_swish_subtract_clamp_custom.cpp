
#include "conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static inline uint32_t ConvtOutDim(uint32_t in, uint32_t stride, uint32_t pad, uint32_t k, uint32_t dil, uint32_t out_pad)
{
    return (in - 1) * stride - 2 * pad + dil * (k - 1) + out_pad + 1;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose3dLogSumExpHardSwishSubtractClampCustomTilingData tiling;

    const auto xShape  = context->GetInputShape(0)->GetStorageShape();
    const auto wShape  = context->GetInputShape(1)->GetStorageShape();
    const auto cbShape = context->GetInputShape(2)->GetStorageShape();
    const auto sbShape = context->GetInputShape(3)->GetStorageShape();

    if (xShape.GetDimNum() != 5 || wShape.GetDimNum() != 5 || cbShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    if (!(sbShape.GetDimNum() == 4 &&
          static_cast<uint32_t>(sbShape.GetDim(0)) == 1 &&
          static_cast<uint32_t>(sbShape.GetDim(1)) == 1 &&
          static_cast<uint32_t>(sbShape.GetDim(2)) == 1 &&
          static_cast<uint32_t>(sbShape.GetDim(3)) == 1)) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t n   = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t cin = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t din = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t hin = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t win = static_cast<uint32_t>(xShape.GetDim(4));

    const uint32_t wcin = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t cout = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kd   = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t kh   = static_cast<uint32_t>(wShape.GetDim(3));
    const uint32_t kw   = static_cast<uint32_t>(wShape.GetDim(4));

    if (wcin != cin) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(cbShape.GetDim(0)) != cout) return ge::GRAPH_FAILED;

    constexpr uint32_t K_EXPECT = 3;
    if (kd != K_EXPECT || kh != K_EXPECT || kw != K_EXPECT) return ge::GRAPH_FAILED;

    // Fixed hyperparams for this benchmark/model:
    constexpr uint32_t STR = 2;
    constexpr uint32_t PAD = 1;
    constexpr uint32_t DIL = 1;
    constexpr uint32_t OUT_PAD = 0;

    const uint32_t dout = ConvtOutDim(din, STR, PAD, kd, DIL, OUT_PAD);
    const uint32_t hout = ConvtOutDim(hin, STR, PAD, kh, DIL, OUT_PAD);
    const uint32_t wout = ConvtOutDim(win, STR, PAD, kw, DIL, OUT_PAD);

    tiling.set_n(n);
    tiling.set_cin(cin);
    tiling.set_din(din);
    tiling.set_hin(hin);
    tiling.set_win(win);

    tiling.set_cout(cout);
    tiling.set_kd(kd);
    tiling.set_kh(kh);
    tiling.set_kw(kw);

    tiling.set_dout(dout);
    tiling.set_hout(hout);
    tiling.set_wout(wout);

    tiling.set_clamp_min(-1.0f);
    tiling.set_clamp_max( 1.0f);

    tiling.set_total_x(static_cast<uint32_t>(xShape.GetShapeSize()));
    tiling.set_total_w(static_cast<uint32_t>(wShape.GetShapeSize()));
    tiling.set_total_conv_b(static_cast<uint32_t>(cbShape.GetShapeSize()));
    tiling.set_total_sub_b(static_cast<uint32_t>(sbShape.GetShapeSize()));
    tiling.set_total_y(static_cast<uint32_t>(context->GetOutputShape(0)->GetStorageShape().GetShapeSize()));

    // Keep BlockDim=1 for stability in this environment; kernel-level overhead was reduced.
    context->SetBlockDim(1);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class ConvTranspose3dLogSumExpHardSwishSubtractClampCustom : public OpDef {
public:
    explicit ConvTranspose3dLogSumExpHardSwishSubtractClampCustom(const char* name) : OpDef(name)
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

        this->Input("sub_bias")
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

OP_ADD(ConvTranspose3dLogSumExpHardSwishSubtractClampCustom);

} // namespace ops
