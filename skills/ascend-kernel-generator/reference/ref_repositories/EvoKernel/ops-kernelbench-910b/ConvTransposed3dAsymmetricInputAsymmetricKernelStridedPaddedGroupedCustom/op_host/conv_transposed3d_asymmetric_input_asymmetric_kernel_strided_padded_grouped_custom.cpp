
#include "conv_transposed3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <stdint.h>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed3dAsymmetricInputAsymmetricKernelStridedPaddedGroupedCustomTilingData tiling;

    // Specialized for the provided model instance.
    constexpr uint32_t totalRows = 8u * 32u * 24u * 48u; // N*Cout*Dout*Hout
    tiling.set_totalRows(totalRows);

    // Tuned: avoid oversubscription that can increase control overhead for scalar-heavy kernels.
    uint32_t blockDim = 224;
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

class ConvTransposed3dAsymmetricInputAsymmetricKernelStridedPaddedGroupedCustom : public OpDef {
public:
    explicit ConvTransposed3dAsymmetricInputAsymmetricKernelStridedPaddedGroupedCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTransposed3dAsymmetricInputAsymmetricKernelStridedPaddedGroupedCustom);

} // namespace ops
