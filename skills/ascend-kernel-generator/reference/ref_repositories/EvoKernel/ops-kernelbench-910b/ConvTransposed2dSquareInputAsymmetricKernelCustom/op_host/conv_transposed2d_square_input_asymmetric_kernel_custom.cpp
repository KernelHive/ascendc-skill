
#include "conv_transposed2d_square_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"
#include <stdint.h>
#include <algorithm>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed2dSquareInputAsymmetricKernelCustomTilingData tiling;

    // Specialized:
    // x: [8,64,512,512], w: [64,64,3,7], stride=1, pad=0, dil=1, outpad=0 => y: [8,64,514,518]
    constexpr uint32_t N = 8;
    constexpr uint32_t COUT = 64;
    constexpr uint32_t HOUT = 514;
    constexpr uint32_t WOUT = 518;

    tiling.set_totalRows(N * COUT * HOUT);
    tiling.set_wout(WOUT);

    // Scalar-heavy kernel benefits from moderate parallelism; keep stable.
    uint32_t blockDim = 256;
    blockDim = std::max(1u, blockDim);
    context->SetBlockDim(blockDim);
    tiling.set_blockDim(blockDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class ConvTransposed2dSquareInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvTransposed2dSquareInputAsymmetricKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTransposed2dSquareInputAsymmetricKernelCustom);

} // namespace ops
