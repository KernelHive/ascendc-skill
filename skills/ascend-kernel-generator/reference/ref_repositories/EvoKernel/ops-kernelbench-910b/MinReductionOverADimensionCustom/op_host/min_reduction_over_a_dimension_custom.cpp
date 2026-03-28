
#include "min_reduction_over_a_dimension_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Specialized to input shape [128, 4096, 4095] and reduce dim=1 (middle axis)
constexpr uint32_t BATCH = 128;
constexpr uint32_t REDUCE_DIM = 4096;
constexpr uint32_t INNER_DIM = 4095;
constexpr uint32_t OUTER_COUNT = BATCH * INNER_DIM;

// Choose a moderate core count; runtime will schedule as appropriate.
// Keep <= typical aicore counts and avoid excessive tiny-per-core work.
constexpr uint32_t BLOCK_DIM = 32;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto inShape = context->GetInputShape(0);
    if (inShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &shape = inShape->GetOriginShape();
    if (shape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint32_t>(shape.GetDim(0)) != BATCH ||
        static_cast<uint32_t>(shape.GetDim(1)) != REDUCE_DIM ||
        static_cast<uint32_t>(shape.GetDim(2)) != INNER_DIM) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputTensor(0)->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    MinReductionOverADimensionCustomTilingData tiling;
    tiling.set_batch(BATCH);
    tiling.set_reduceDim(REDUCE_DIM);
    tiling.set_innerDim(INNER_DIM);
    tiling.set_outerCount(OUTER_COUNT);

    const uint32_t colsPerCore = (OUTER_COUNT + BLOCK_DIM - 1) / BLOCK_DIM;
    tiling.set_colsPerCore(colsPerCore);

    context->SetTilingKey(1);
    context->SetBlockDim(BLOCK_DIM);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    // Input: [128,4096,4095], reduce dim=1 => output [128,4095]
    gert::Shape *out = context->GetOutputShape(0);
    *out = {optiling::BATCH, optiling::INNER_DIM};
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto dt = context->GetInputDataType(0);
    context->SetOutputDataType(0, dt);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class MinReductionOverADimensionCustom : public OpDef {
public:
    explicit MinReductionOverADimensionCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
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

OP_ADD(MinReductionOverADimensionCustom);

} // namespace ops
