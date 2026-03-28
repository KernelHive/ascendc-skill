
#include "layer_norm_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Prefer higher occupancy for small batch by allowing more blocks, but keep conservative.
static constexpr uint32_t BLOCK_DIM_MAX = 24;
static constexpr uint32_t TILE_LENGTH = 4096;
static constexpr float EPS = 1.0e-5f;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    LayerNormCustomTilingData tiling;

    const auto* inShape = context->GetInputShape(0);
    const auto& s = inShape->GetOriginShape();
    const size_t rank = s.GetDimNum();

    uint64_t cols64 = 0;
    uint64_t rows64 = 0;

    if (rank < 3) {
        cols64 = 0;
        rows64 = 0;
    } else {
        cols64 = 1;
        cols64 *= static_cast<uint64_t>(s.GetDim(rank - 3));
        cols64 *= static_cast<uint64_t>(s.GetDim(rank - 2));
        cols64 *= static_cast<uint64_t>(s.GetDim(rank - 1));

        rows64 = 1;
        for (size_t i = 0; i + 3 < rank; ++i) {
            rows64 *= static_cast<uint64_t>(s.GetDim(i));
        }
        if (rank == 3) rows64 = 1;
    }

    if (cols64 > 0xFFFFFFFFULL) cols64 = 0xFFFFFFFFULL;
    if (rows64 > 0xFFFFFFFFULL) rows64 = 0xFFFFFFFFULL;
    const uint32_t cols = static_cast<uint32_t>(cols64);
    const uint32_t rows = static_cast<uint32_t>(rows64);

    uint32_t blockDim = BLOCK_DIM_MAX;
    if (rows > 0 && rows < blockDim) blockDim = rows;
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    tiling.set_rows(rows);
    tiling.set_cols(cols);
    tiling.set_tileLength(TILE_LENGTH);
    tiling.set_eps(EPS);

    float invCols = 0.0f;
    if (cols != 0u) invCols = 1.0f / static_cast<float>(cols);
    tiling.set_invCols(invCols);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto dt = context->GetInputDataType(0);
    context->SetOutputDataType(0, dt);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class LayerNormCustom : public OpDef {
public:
    explicit LayerNormCustom(const char* name) : OpDef(name)
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

OP_ADD(LayerNormCustom);

}  // namespace ops
