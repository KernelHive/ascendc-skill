
#include "cross_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <cmath>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    // q: [B,H,Sq,D], k/v: [B,H,Sk,D]
    auto qShape = context->GetInputShape(0)->GetStorageShape();
    auto kShape = context->GetInputShape(1)->GetStorageShape();
    auto vShape = context->GetInputShape(2)->GetStorageShape();

    if (qShape.GetDimNum() != 4 || kShape.GetDimNum() != 4 || vShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t b  = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t h  = static_cast<uint32_t>(qShape.GetDim(1));
    const uint32_t sq = static_cast<uint32_t>(qShape.GetDim(2));
    const uint32_t d  = static_cast<uint32_t>(qShape.GetDim(3));

    const uint32_t bk = static_cast<uint32_t>(kShape.GetDim(0));
    const uint32_t hk = static_cast<uint32_t>(kShape.GetDim(1));
    const uint32_t sk = static_cast<uint32_t>(kShape.GetDim(2));
    const uint32_t dk = static_cast<uint32_t>(kShape.GetDim(3));

    const uint32_t bv = static_cast<uint32_t>(vShape.GetDim(0));
    const uint32_t hv = static_cast<uint32_t>(vShape.GetDim(1));
    const uint32_t sv = static_cast<uint32_t>(vShape.GetDim(2));
    const uint32_t dv = static_cast<uint32_t>(vShape.GetDim(3));

    if (b == 0 || h == 0 || sq == 0 || sk == 0 || d == 0) return ge::GRAPH_FAILED;

    if (bk != b || hk != h || dk != d) return ge::GRAPH_FAILED;
    if (bv != b || hv != h || sv != sk || dv != d) return ge::GRAPH_FAILED;

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) return ge::GRAPH_FAILED;
    const float* scalePtr = attrs->GetAttrPointer<float>(0);
    if (scalePtr == nullptr) return ge::GRAPH_FAILED;

    const float scale = *scalePtr;
    if (!std::isfinite(scale)) return ge::GRAPH_FAILED;

    CrossAttentionCustomTilingData tiling;
    tiling.set_b(b);
    tiling.set_h(h);
    tiling.set_sq(sq);
    tiling.set_sk(sk);
    tiling.set_d(d);
    tiling.set_scale(scale);

    // One core per (b,h)
    const uint32_t block_dim = b * h;
    if (block_dim == 0) return ge::GRAPH_FAILED;
    context->SetBlockDim(block_dim);

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
    const auto* qShape = context->GetInputShape(0);
    if (outShape == nullptr || qShape == nullptr) return GRAPH_FAILED;
    *outShape = *qShape; // [B,H,Sq,D]
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class CrossAttentionCustom : public OpDef {
public:
    explicit CrossAttentionCustom(const char* name) : OpDef(name)
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

OP_ADD(CrossAttentionCustom);
} // namespace ops
