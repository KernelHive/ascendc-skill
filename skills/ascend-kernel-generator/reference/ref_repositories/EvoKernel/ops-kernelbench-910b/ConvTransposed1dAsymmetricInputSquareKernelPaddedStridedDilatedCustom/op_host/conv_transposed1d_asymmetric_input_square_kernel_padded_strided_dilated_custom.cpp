
#include "conv_transposed1d_asymmetric_input_square_kernel_padded_strided_dilated_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustomTilingData tiling;

    constexpr uint32_t N = 16;
    constexpr uint32_t CIN = 32;
    constexpr uint32_t COUT = 64;
    constexpr uint32_t LIN = 131072;
    constexpr uint32_t LOUT = 262145;

    tiling.set_n(N);
    tiling.set_cin(CIN);
    tiling.set_cout(COUT);
    tiling.set_lin(LIN);
    tiling.set_lout(LOUT);

    // One block per (n, co) to remove div/mod over Lout and improve linearity.
    // 16*64 = 1024 blocks provides high parallelism while keeping per-block work large.
    context->SetBlockDim(N * COUT);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustom : public OpDef {
public:
    explicit ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustom(const char* name) : OpDef(name)
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

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustom);

} // namespace ops
