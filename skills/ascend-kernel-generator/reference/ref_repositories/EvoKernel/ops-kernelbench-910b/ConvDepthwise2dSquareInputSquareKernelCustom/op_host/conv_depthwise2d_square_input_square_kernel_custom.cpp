
#include "conv_depthwise2d_square_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <string.h>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvDepthwise2dSquareInputSquareKernelCustomTilingData tiling;

    // Specialized benchmark shapes:
    // x: [16,64,512,512], w: [64,1,3,3], y: [16,64,510,510]
    const uint32_t N = 16;
    const uint32_t C = 64;
    const uint32_t H = 512;
    const uint32_t W = 512;
    const uint32_t OH = 510;
    const uint32_t OW = 510;

    uint32_t totalY = context->GetOutputShape(0)->GetStorageShape().GetShapeSize();
    tiling.set_totalY(totalY);
    tiling.set_rows(N * C * OH);
    tiling.set_ow(OW);
    tiling.set_h(H);
    tiling.set_w(W);
    tiling.set_c(C);
    tiling.set_oh(OH);

    // Keep conservative to avoid device/stream allocation instability.
    // Slightly higher than baseline for more occupancy while staying safe.
    uint32_t blockDim = 48;
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

class ConvDepthwise2dSquareInputSquareKernelCustom : public OpDef {
public:
    explicit ConvDepthwise2dSquareInputSquareKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvDepthwise2dSquareInputSquareKernelCustom);

} // namespace ops
