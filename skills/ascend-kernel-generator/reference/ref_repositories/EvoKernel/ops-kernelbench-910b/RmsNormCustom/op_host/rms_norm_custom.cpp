
#include "rms_norm_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <algorithm>

namespace optiling {

// More blocks to improve occupancy; map blocks across (b, innerTile).
static constexpr uint32_t MAX_BLOCK_DIM = 48;
// Keep UB usage modest; contiguous along inner for good MTE bursts.
static constexpr uint32_t INNER_TILE = 256;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    RmsNormCustomTilingData tiling;

    constexpr float EPS = 1e-5f;

    const auto* inShape = context->GetInputShape(0);
    const auto& s = inShape->GetOriginShape();
    const size_t rank = s.GetDimNum();

    uint32_t b = 0;
    uint32_t c = 0;
    uint32_t inner = 0;

    if (rank >= 2) {
        b = static_cast<uint32_t>(s.GetDim(0));
        c = static_cast<uint32_t>(s.GetDim(1));
        uint64_t inner64 = 1;
        for (size_t i = 2; i < rank; ++i) {
            inner64 *= static_cast<uint64_t>(s.GetDim(i));
        }
        if (inner64 > 0xFFFFFFFFULL) inner64 = 0xFFFFFFFFULL;
        inner = static_cast<uint32_t>(inner64);
    }

    const uint32_t innerTile = (inner == 0) ? 0 : std::min<uint32_t>(INNER_TILE, inner);
    const uint32_t tilesPerB = (innerTile == 0) ? 0 : (inner + innerTile - 1u) / innerTile;

    // BlockDim is capped; each block will take a strided subset of tiles.
    uint64_t totalTiles64 = static_cast<uint64_t>(b) * static_cast<uint64_t>(tilesPerB);
    if (totalTiles64 == 0) totalTiles64 = 1;
    uint32_t blockDim = std::min<uint32_t>(MAX_BLOCK_DIM, static_cast<uint32_t>(std::min<uint64_t>(totalTiles64, 0xFFFFFFFFULL)));
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    tiling.set_b(b);
    tiling.set_c(c);
    tiling.set_inner(inner);
    tiling.set_innerTile(innerTile);
    tiling.set_tilesPerB(tilesPerB);
    tiling.set_eps(EPS);

    float invC = 0.0f;
    if (c != 0) invC = 1.0f / static_cast<float>(c);
    tiling.set_invC(invC);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
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
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class RmsNormCustom : public OpDef {
public:
    explicit RmsNormCustom(const char* name) : OpDef(name)
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

OP_ADD(RmsNormCustom);

}  // namespace ops
