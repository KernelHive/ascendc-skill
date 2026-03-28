
#include "argmin_over_a_dimension_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Specialized fixed contract:
// x: [128,4096,4095] float32, reduce dim=1 -> y: [128,4095] int64
static constexpr uint32_t BATCH = 128;
static constexpr uint32_t REDUCE_DIM = 4096;
static constexpr uint32_t INNER_DIM = 4095;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ArgminOverADimensionCustomTilingData tiling;

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

    if (context->GetInputTensor(0) == nullptr ||
        context->GetInputTensor(0)->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    uint32_t totalX = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t totalY = context->GetOutputShape(0)->GetStorageShape().GetShapeSize();

    tiling.set_totalX(totalX);
    tiling.set_totalY(totalY);
    tiling.set_batch(BATCH);
    tiling.set_reduceDim(REDUCE_DIM);
    tiling.set_innerDim(INNER_DIM);

    // Unroll reduction; 4096 divisible by 8.
    tiling.set_unrollR(8);

    // Let runtime decide effective core count; set a safe upper bound.
    // Using 256 is common on 910B; runtime will clamp as needed.
    context->SetBlockDim(256);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class ArgminOverADimensionCustom : public OpDef {
public:
    explicit ArgminOverADimensionCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ArgminOverADimensionCustom);

} // namespace ops
