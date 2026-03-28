
#include "external_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    auto xShape = context->GetInputShape(0)->GetStorageShape();
    if (xShape.GetDimNum() != 3) return ge::GRAPH_FAILED;

    const uint32_t bs = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t n  = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t s  = static_cast<uint32_t>(xShape.GetDim(2));
    if (bs == 0 || n == 0 || s == 0) return ge::GRAPH_FAILED;

    // Keep specialization for benchmark-friendly S=64 and N<=49.
    // (Note: generalizing S would require different UB planning and vector widths.)
    if (s != 64 || n > 49) return ge::GRAPH_FAILED;

    ExternalAttentionCustomTilingData tiling;
    tiling.set_bs(bs);
    tiling.set_n(n);
    tiling.set_s(s);

    // One core per batch: no cross-core dependency for softmax(dim=1).
    context->SetBlockDim(bs);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    auto* outShape = context->GetOutputShape(0);
    const auto* inShape = context->GetInputShape(0);
    if (outShape == nullptr || inShape == nullptr) return GRAPH_FAILED;
    *outShape = *inShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class ExternalAttentionCustom : public OpDef {
public:
    explicit ExternalAttentionCustom(const char* name) : OpDef(name)
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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ExternalAttentionCustom);
} // namespace ops
