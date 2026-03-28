
#include "elu_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>

namespace optiling {

// Keep concurrency suitable for 910B, but make it shape-aware to avoid oversubscription on tiny tensors.
static constexpr uint32_t kDefaultBlockDim = 24;

// 8192 fp32 elems = 32KB; double-buffer in/out + tmp fits well and improves MTE burst efficiency.
static constexpr uint32_t kTileElems = 8192;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    EluCustomTilingData tiling;
    const uint32_t totalLength =
        static_cast<uint32_t>(context->GetInputShape(0)->GetOriginShape().GetShapeSize());

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *alpha = attrs->GetAttrPointer<float>(0);

    uint32_t blockDim = kDefaultBlockDim;
    if (totalLength == 0) blockDim = 1;
    if (totalLength < blockDim) blockDim = 1;

    context->SetBlockDim(blockDim);

    tiling.set_totalLength(totalLength);
    tiling.set_blockDim(blockDim);
    tiling.set_tileElems(kTileElems);
    tiling.set_alpha(*alpha);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x_dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class EluCustom : public OpDef {
public:
    explicit EluCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("alpha").AttrType(OPTIONAL).Float(1.0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(EluCustom);
} // namespace ops
