
#include "sum_reduction_over_a_dimension_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;

    const auto origin = context->GetInputShape(0)->GetOriginShape();
    const int64_t rank = static_cast<int64_t>(origin.GetDimNum());
    if (rank != 3) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t B = static_cast<uint32_t>(origin.GetDim(0));
    const uint32_t N = static_cast<uint32_t>(origin.GetDim(1));
    const uint32_t S = static_cast<uint32_t>(origin.GetDim(2));

    const int64_t *dimPtr = context->GetAttrs()->GetInt(0);
    if (dimPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t dimAttr = *dimPtr;
    if (dimAttr < 0) dimAttr += rank;
    if (dimAttr != 1) { // specialized for reduce along N
        return ge::GRAPH_FAILED;
    }

    tiling.set_B(B);
    tiling.set_N(N);
    tiling.set_S(S);
    tiling.set_outerCount(B * S);

    context->SetTilingKey(0);

    // Fixed launch for this specialized workload to reduce variability and mapping overhead.
    // Runtime will cap if needed.
    constexpr uint32_t kBlockDim = 24;
    context->SetBlockDim(kBlockDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inShape = context->GetInputShape(0);
    if (inShape == nullptr) return GRAPH_FAILED;
    if (inShape->GetDimNum() != 3) return GRAPH_FAILED;

    const int64_t B = inShape->GetDim(0);
    const int64_t S = inShape->GetDim(2);

    gert::Shape *outShape = context->GetOutputShape(0);
    *outShape = gert::Shape({B, 1, S});
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class SumReductionOverADimensionCustom : public OpDef {
public:
    explicit SumReductionOverADimensionCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->Attr("dim").Int();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(SumReductionOverADimensionCustom);
} // namespace ops
