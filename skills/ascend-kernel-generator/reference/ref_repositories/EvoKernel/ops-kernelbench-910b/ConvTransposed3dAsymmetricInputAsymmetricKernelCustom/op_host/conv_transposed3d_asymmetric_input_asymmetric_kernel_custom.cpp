
#include "conv_transposed3d_asymmetric_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"
#include <stdint.h>
#include <algorithm>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed3dAsymmetricInputAsymmetricKernelCustomTilingData tiling;

    uint32_t totalX = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t totalW = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    uint32_t totalY = context->GetOutputShape(0)->GetStorageShape().GetShapeSize();

    tiling.set_totalX(totalX);
    tiling.set_totalW(totalW);
    tiling.set_totalY(totalY);

    // Conservative increase in parallelism (avoid very high blockDim instability).
    uint32_t blockDim = 48;
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

class ConvTransposed3dAsymmetricInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvTransposed3dAsymmetricInputAsymmetricKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTransposed3dAsymmetricInputAsymmetricKernelCustom);

} // namespace ops
