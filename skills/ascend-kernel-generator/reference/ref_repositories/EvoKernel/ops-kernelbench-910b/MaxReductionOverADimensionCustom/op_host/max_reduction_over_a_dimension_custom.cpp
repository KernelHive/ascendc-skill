
#include "max_reduction_over_a_dimension_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Specialized fixed contract for this kernel:
// x: [128,4096,4095] float32, reduce dim=1 -> y: [128,4095]
static constexpr uint32_t BATCH = 128;
static constexpr uint32_t REDUCE_DIM = 4096;
static constexpr uint32_t INNER_DIM = 4095;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MaxReductionOverADimensionCustomTilingData tiling;

    auto inShape = context->GetInputShape(0);
    if (inShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &shape = inShape->GetOriginShape();
    if (shape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    if ((uint32_t)shape.GetDim(0) != BATCH ||
        (uint32_t)shape.GetDim(1) != REDUCE_DIM ||
        (uint32_t)shape.GetDim(2) != INNER_DIM) {
        return ge::GRAPH_FAILED;
    }

    uint32_t totalX = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t totalY = context->GetOutputShape(0)->GetStorageShape().GetShapeSize();

    tiling.set_totalX(totalX);
    tiling.set_totalY(totalY);
    tiling.set_batch(BATCH);
    tiling.set_reduceDim(REDUCE_DIM);
    tiling.set_innerDim(INNER_DIM);

    // One block per batch item (stable mapping, predictable coverage).
    context->SetBlockDim(BATCH);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class MaxReductionOverADimensionCustom : public OpDef {
public:
    explicit MaxReductionOverADimensionCustom(const char* name) : OpDef(name)
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

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MaxReductionOverADimensionCustom);

} // namespace ops
