
#include "conv_depthwise2d_asymmetric_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"
#include <string.h>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvDepthwise2dAsymmetricInputAsymmetricKernelCustomTilingData tiling;

    // Specialized for benchmark:
    // x: [N=32, C=128, H=128, W=256]
    // weight: [C=128, 1, Kh=3, Kw=7] (depthwise)
    // stride=1, pad=0, dilation=1 => Ho=126, Wo=250
    // y: [32, 128, 126, 250]
    uint32_t totalX = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t totalW = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    uint32_t totalY = context->GetOutputShape(0)->GetStorageShape().GetShapeSize();

    tiling.set_totalX(totalX);
    tiling.set_totalW(totalW);
    tiling.set_totalY(totalY);

    // Conservative single core baseline.
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

class ConvDepthwise2dAsymmetricInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvDepthwise2dAsymmetricInputAsymmetricKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvDepthwise2dAsymmetricInputAsymmetricKernelCustom);

} // namespace ops
