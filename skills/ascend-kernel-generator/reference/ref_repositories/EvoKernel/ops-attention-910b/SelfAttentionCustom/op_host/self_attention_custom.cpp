
#include "self_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    auto qShape = context->GetInputShape(0)->GetStorageShape();
    auto kShape = context->GetInputShape(1)->GetStorageShape();
    auto vShape = context->GetInputShape(2)->GetStorageShape();

    if (qShape.GetDimNum() != 4 || kShape.GetDimNum() != 4 || vShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t b = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t h = static_cast<uint32_t>(qShape.GetDim(1));
    const uint32_t s = static_cast<uint32_t>(qShape.GetDim(2));
    const uint32_t d = static_cast<uint32_t>(qShape.GetDim(3));
    if (b == 0 || h == 0 || s == 0 || d == 0) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(kShape.GetDim(0)) != b || static_cast<uint32_t>(kShape.GetDim(1)) != h ||
        static_cast<uint32_t>(kShape.GetDim(2)) != s || static_cast<uint32_t>(kShape.GetDim(3)) != d) {
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint32_t>(vShape.GetDim(0)) != b || static_cast<uint32_t>(vShape.GetDim(1)) != h ||
        static_cast<uint32_t>(vShape.GetDim(2)) != s || static_cast<uint32_t>(vShape.GetDim(3)) != d) {
        return ge::GRAPH_FAILED;
    }

    SelfAttentionCustomTilingData tiling;
    tiling.set_b(b);
    tiling.set_h(h);
    tiling.set_s(s);
    tiling.set_d(d);
    tiling.set_scale(1.0f / std::sqrt(static_cast<float>(d)));

    const uint64_t totalRows = static_cast<uint64_t>(b) * h * s;
    const uint32_t blockDim = static_cast<uint32_t>(std::min<uint64_t>(totalRows, 65535ull));
    if (blockDim == 0) return ge::GRAPH_FAILED;
    context->SetBlockDim(blockDim);

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
    *outShape = *qShape; // [B,H,S,D]
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) return GRAPH_FAILED;
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class SelfAttentionCustom : public OpDef {
public:
    explicit SelfAttentionCustom(const char* name) : OpDef(name)
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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SelfAttentionCustom);
} // namespace ops
