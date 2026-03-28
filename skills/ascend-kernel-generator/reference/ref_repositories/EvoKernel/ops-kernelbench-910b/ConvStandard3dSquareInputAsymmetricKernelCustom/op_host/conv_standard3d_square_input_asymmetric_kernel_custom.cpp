
#include "conv_standard3d_square_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <string.h>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvStandard3dSquareInputAsymmetricKernelCustomTilingData tiling;

    uint32_t totalX = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t totalW = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    uint32_t totalY = context->GetOutputShape(0)->GetStorageShape().GetShapeSize();

    tiling.set_totalX(totalX);
    tiling.set_totalW(totalW);
    tiling.set_totalY(totalY);

    // More blocks to reduce pipeline gaps; work per block is uniform (pad=0,stride=1).
    // Keep moderate to avoid overhead/instability.
    uint32_t blockDim = 192;
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

class ConvStandard3dSquareInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvStandard3dSquareInputAsymmetricKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvStandard3dSquareInputAsymmetricKernelCustom);

} // namespace ops
