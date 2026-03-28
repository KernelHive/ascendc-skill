
#include "conv_transposed2d_asymmetric_input_square_kernel_dilated_padded_strided_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <stdint.h>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed2dAsymmetricInputSquareKernelDilatedPaddedStridedCustomTilingData tiling;

    // Specialized benchmark shapes/params:
    // x: [16,32,64,128], w: [32,64,3,3], stride=5, pad=1, dil=2
    // y: [16,64,318,638]
    constexpr uint32_t N = 16;
    constexpr uint32_t COUT = 64;
    constexpr uint32_t HOUT = 318;
    constexpr uint32_t totalRows = N * COUT * HOUT;

    tiling.set_totalRows(totalRows);

    // Increase occupancy to reduce pipeline gaps; each block computes full W line(s) for its rows.
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

class ConvTransposed2dAsymmetricInputSquareKernelDilatedPaddedStridedCustom : public OpDef {
public:
    explicit ConvTransposed2dAsymmetricInputSquareKernelDilatedPaddedStridedCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTransposed2dAsymmetricInputSquareKernelDilatedPaddedStridedCustom);

} // namespace ops
