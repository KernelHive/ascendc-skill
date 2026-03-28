
#include "conv_transposed2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated_custom_tiling.h"
#include "register/op_def_registry.h"
#include <string.h>
#include <algorithm>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed2dAsymmetricInputAsymmetricKernelStridedGroupedPaddedDilatedCustomTilingData tiling;

    uint32_t totalX = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t totalW = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    uint32_t totalY = context->GetOutputShape(0)->GetStorageShape().GetShapeSize();

    tiling.set_totalX(totalX);
    tiling.set_totalW(totalW);
    tiling.set_totalY(totalY);

    // Expose parallelism over output elements; keep moderate to avoid device/context stress.
    // 24 blocks generally gives good occupancy on 910B without extreme launch configurations.
    uint32_t blockDim = 24;
    tiling.set_blockDim(blockDim);

    // Each block processes y in chunks; tune for less loop overhead but keep register pressure modest.
    uint32_t tileElems = 256;
    tiling.set_tileElems(tileElems);

    context->SetBlockDim(blockDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class ConvTransposed2dAsymmetricInputAsymmetricKernelStridedGroupedPaddedDilatedCustom : public OpDef {
public:
    explicit ConvTransposed2dAsymmetricInputAsymmetricKernelStridedGroupedPaddedDilatedCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTransposed2dAsymmetricInputAsymmetricKernelStridedGroupedPaddedDilatedCustom);

} // namespace ops
