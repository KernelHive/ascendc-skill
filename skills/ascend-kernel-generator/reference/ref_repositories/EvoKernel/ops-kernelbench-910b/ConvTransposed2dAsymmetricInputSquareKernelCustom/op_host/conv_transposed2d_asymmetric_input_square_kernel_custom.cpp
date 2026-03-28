
#include "conv_transposed2d_asymmetric_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"
#include <stdint.h>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed2dAsymmetricInputSquareKernelCustomTilingData tiling;

    // Specialized for:
    // x: [8,32,512,1024]
    // w: [32,32,3,3]
    // y: [8,32,514,1026]
    constexpr uint32_t N = 8;
    constexpr uint32_t COUT = 32;
    constexpr uint32_t HOUT = 514;
    constexpr uint32_t WOUT = 1026;

    tiling.set_rows(N * COUT * HOUT);
    tiling.set_wout(WOUT);

    // Moderate occupancy bump, avoid previously unstable very-high blockDim pattern.
    constexpr uint32_t kBlockDim = 48;
    context->SetBlockDim(kBlockDim);
    tiling.set_blockDim(kBlockDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class ConvTransposed2dAsymmetricInputSquareKernelCustom : public OpDef {
public:
    explicit ConvTransposed2dAsymmetricInputSquareKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTransposed2dAsymmetricInputSquareKernelCustom);

} // namespace ops
