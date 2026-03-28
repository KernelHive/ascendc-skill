
#include "argmax_over_a_dimension_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Keep specialization contract (stable and fastest for the known model),
// but launch many more blocks safely by tiling innerDim.
// 4095 / 64 => 64 tiles per batch => 8192 blocks total.
static constexpr uint32_t BATCH = 128;
static constexpr uint32_t REDUCE_DIM = 4096;
static constexpr uint32_t INNER_DIM = 4095;
static constexpr uint32_t TILE_INNER = 64;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ArgmaxOverADimensionCustomTilingData tiling;

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
    tiling.set_tileInner(TILE_INNER);

    const uint32_t tilesPerBatch = (INNER_DIM + TILE_INNER - 1) / TILE_INNER;
    tiling.set_tilesPerBatch(tilesPerBatch);

    // 1 block per (b, tile).
    uint32_t blockDim = BATCH * tilesPerBatch;
    context->SetBlockDim(blockDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class ArgmaxOverADimensionCustom : public OpDef {
public:
    explicit ArgmaxOverADimensionCustom(const char* name) : OpDef(name)
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

OP_ADD(ArgmaxOverADimensionCustom);

} // namespace ops
