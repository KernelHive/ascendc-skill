
#include "masked_cumsum_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <climits>

namespace optiling {

static inline uint32_t SafeU32(int64_t v)
{
    if (v <= 0) return 0;
    if (v > static_cast<int64_t>(UINT32_MAX)) return UINT32_MAX;
    return static_cast<uint32_t>(v);
}

static inline int64_t NormalizeDim(int64_t dim, int64_t rank)
{
    if (rank <= 0) return 0;
    int64_t d = dim;
    if (d < 0) d += rank;
    if (d < 0) d = 0;
    if (d >= rank) d = rank - 1;
    return d;
}

// Safe occupancy bump without being too aggressive.
static constexpr uint32_t BLOCK_DIM = 96;

// UB budget: double-buffer x/y + double-buffer mask
// Choose 4096 elems by default; can be tuned if needed.
static constexpr uint32_t TILE_ELEMS = 4096;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;

    const auto shape = context->GetInputShape(0)->GetOriginShape();
    const int64_t rank64 = static_cast<int64_t>(shape.GetDimNum());
    const uint32_t rank = SafeU32(rank64 <= 0 ? 1 : rank64);

    int64_t dimAttrVal = 0;
    const int64_t *dimPtr = context->GetAttrs()->GetInt(0);
    if (dimPtr != nullptr) dimAttrVal = *dimPtr;

    const int64_t dimNorm64 = NormalizeDim(dimAttrVal, rank64 <= 0 ? 1 : rank64);

    uint64_t total64 = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        uint64_t d = static_cast<uint64_t>(shape.GetDim(i));
        total64 *= d;
    }

    uint64_t axis64 = 0;
    if (rank64 > 0) {
        axis64 = static_cast<uint64_t>(shape.GetDim(static_cast<size_t>(dimNorm64)));
    } else {
        axis64 = total64;
    }
    uint64_t outer64 = (axis64 == 0) ? 0 : (total64 / axis64);

    context->SetBlockDim(BLOCK_DIM);
    context->SetTilingKey(0);

    tiling.set_rank(rank);
    tiling.set_dim(static_cast<int32_t>(dimNorm64));
    tiling.set_outerSize(SafeU32(static_cast<int64_t>(outer64)));
    tiling.set_axisSize(SafeU32(static_cast<int64_t>(axis64)));
    tiling.set_totalElems(SafeU32(static_cast<int64_t>(total64)));
    tiling.set_tileElems(TILE_ELEMS);

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
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class MaskedCumsumCustom : public OpDef {
public:
    explicit MaskedCumsumCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
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

OP_ADD(MaskedCumsumCustom);

} // namespace ops
