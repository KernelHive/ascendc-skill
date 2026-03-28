
#include "scaled_dot_product_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <cmath>

namespace optiling {

// Device kernel fast-path limits (UB is statically sized around these).
static constexpr uint32_t MAX_S = 128;
static constexpr uint32_t MAX_D = 64;

// Conservative cap to avoid too many blocks; for typical shapes S<=128 this is plenty.
static constexpr uint32_t MAX_TASKS_PER_BH = 8;

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

    // Enforce fast-path constraints (kernel UB is fixed-size).
    if (s > MAX_S || d > MAX_D) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(kShape.GetDim(0)) != b || static_cast<uint32_t>(kShape.GetDim(1)) != h ||
        static_cast<uint32_t>(kShape.GetDim(2)) != s || static_cast<uint32_t>(kShape.GetDim(3)) != d) {
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint32_t>(vShape.GetDim(0)) != b || static_cast<uint32_t>(vShape.GetDim(1)) != h ||
        static_cast<uint32_t>(vShape.GetDim(2)) != s) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) return ge::GRAPH_FAILED;
    const int64_t* dkPtr = attrs->GetAttrPointer<int64_t>(0);
    if (dkPtr == nullptr) return ge::GRAPH_FAILED;
    const int64_t d_k = *dkPtr;
    if (d_k <= 0) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(d_k) != d) return ge::GRAPH_FAILED;

    // Fused kernel constraint d_v == d_k for output [B,H,S,D]
    const uint32_t dV = static_cast<uint32_t>(vShape.GetDim(3));
    if (dV != d) return ge::GRAPH_FAILED;

    // Parallelize over query rows within each (b,h): split S rows into tasksPerBH parts.
    uint32_t tasksPerBH = 1;
    if (s >= 64) tasksPerBH = 4;
    else if (s >= 32) tasksPerBH = 2;
    if (tasksPerBH > MAX_TASKS_PER_BH) tasksPerBH = MAX_TASKS_PER_BH;
    if (tasksPerBH > s) tasksPerBH = s;
    if (tasksPerBH == 0) return ge::GRAPH_FAILED;

    ScaledDotProductAttentionCustomTilingData tiling;
    tiling.set_b(b);
    tiling.set_h(h);
    tiling.set_s(s);
    tiling.set_d(d);
    tiling.set_scale(1.0f / std::sqrt(static_cast<float>(d)));
    tiling.set_tasksPerBH(tasksPerBH);

    const uint32_t block_dim = b * h * tasksPerBH;
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
    *outShape = *qShape; // output [B,H,S,D]
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class ScaledDotProductAttentionCustom : public OpDef {
public:
    explicit ScaledDotProductAttentionCustom(const char* name) : OpDef(name)
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

        this->Attr("d_k").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ScaledDotProductAttentionCustom);
} // namespace ops
