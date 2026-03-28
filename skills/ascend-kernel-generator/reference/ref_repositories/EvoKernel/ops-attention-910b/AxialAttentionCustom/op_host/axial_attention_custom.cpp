
#include "axial_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <cmath>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    // q,k,v: [BH, T, E]
    auto qShape = context->GetInputShape(0)->GetStorageShape();
    auto kShape = context->GetInputShape(1)->GetStorageShape();
    auto vShape = context->GetInputShape(2)->GetStorageShape();

    if (qShape.GetDimNum() != 3 || kShape.GetDimNum() != 3 || vShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t bh = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t t  = static_cast<uint32_t>(qShape.GetDim(1));
    const uint32_t e  = static_cast<uint32_t>(qShape.GetDim(2));

    const uint32_t bkh = static_cast<uint32_t>(kShape.GetDim(0));
    const uint32_t tk  = static_cast<uint32_t>(kShape.GetDim(1));
    const uint32_t ek  = static_cast<uint32_t>(kShape.GetDim(2));

    const uint32_t bvh = static_cast<uint32_t>(vShape.GetDim(0));
    const uint32_t tv  = static_cast<uint32_t>(vShape.GetDim(1));
    const uint32_t ev  = static_cast<uint32_t>(vShape.GetDim(2));

    if (bh == 0 || t == 0 || e == 0) return ge::GRAPH_FAILED;
    if (bkh != bh || tk != t || ek != e) return ge::GRAPH_FAILED;
    if (bvh != bh || tv != t || ev != e) return ge::GRAPH_FAILED;

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) return ge::GRAPH_FAILED;
    const float* scalePtr = attrs->GetAttrPointer<float>(0);
    if (scalePtr == nullptr) return ge::GRAPH_FAILED;
    const float scale = *scalePtr;
    if (!std::isfinite(scale)) return ge::GRAPH_FAILED;

    AxialAttentionCustomTilingData tiling;
    tiling.set_bh(bh);
    tiling.set_t(t);
    tiling.set_e(e);
    tiling.set_scale(scale);

    // One core per BH row
    context->SetBlockDim(bh);

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
    if (context == nullptr) return GRAPH_FAILED;
    auto* outShape = context->GetOutputShape(0);
    const auto* qShape = context->GetInputShape(0);
    if (outShape == nullptr || qShape == nullptr) return GRAPH_FAILED;
    *outShape = *qShape; // [BH,T,E]
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class AxialAttentionCustom : public OpDef {
public:
    explicit AxialAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("scale").Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AxialAttentionCustom);
} // namespace ops
