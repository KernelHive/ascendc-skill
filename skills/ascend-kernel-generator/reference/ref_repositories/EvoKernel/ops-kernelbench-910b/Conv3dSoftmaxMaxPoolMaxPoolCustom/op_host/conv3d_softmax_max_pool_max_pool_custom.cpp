
#include "conv3d_softmax_max_pool_max_pool_custom_tiling.h"
#include "register/op_def_registry.h"
#include <string.h>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv3dSoftmaxMaxPoolMaxPoolCustomTilingData tiling;

    uint32_t totalX = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t totalW = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    uint32_t totalB = context->GetInputShape(2)->GetStorageShape().GetShapeSize();
    uint32_t totalY = context->GetOutputShape(0)->GetStorageShape().GetShapeSize();

    tiling.set_totalX(totalX);
    tiling.set_totalW(totalW);
    tiling.set_totalB(totalB);
    tiling.set_totalY(totalY);

    // Keep stable configuration; performance comes from kernel vectorization.
    context->SetBlockDim(32);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class Conv3dSoftmaxMaxPoolMaxPoolCustom : public OpDef {
public:
    explicit Conv3dSoftmaxMaxPoolMaxPoolCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("weight_packed")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("bias")
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

OP_ADD(Conv3dSoftmaxMaxPoolMaxPoolCustom);

} // namespace ops
