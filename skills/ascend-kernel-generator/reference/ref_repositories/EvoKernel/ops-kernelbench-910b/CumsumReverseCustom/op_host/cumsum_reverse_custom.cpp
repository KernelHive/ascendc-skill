
#include "cumsum_reverse_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <algorithm>

namespace optiling {

static constexpr uint32_t MAX_BLOCK_DIM = 256;
static constexpr uint32_t MIN_BLOCK_DIM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;

    const auto shape = context->GetInputShape(0)->GetOriginShape();
    const uint32_t rank = static_cast<uint32_t>(shape.GetDimNum());
    const uint32_t totalElems = static_cast<uint32_t>(shape.GetShapeSize());

    // Specialized: 2D ND tensor, reverse cumsum along last dimension.
    if (rank != 2) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t rows = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t cols = static_cast<uint32_t>(shape.GetDim(1));

    uint32_t blockDim = 1;
    if (rows > 0) {
        blockDim = std::min(rows, MAX_BLOCK_DIM);
        blockDim = std::max(blockDim, MIN_BLOCK_DIM);
        // If rows < MIN, keep blockDim=rows (avoid launching more blocks than work).
        if (rows < MIN_BLOCK_DIM) blockDim = rows;
        if (blockDim == 0) blockDim = 1;
    }

    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);

    tiling.set_rows(rows);
    tiling.set_cols(cols);
    tiling.set_totalElems(totalElems);
    tiling.set_blockDim(blockDim);

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
class CumsumReverseCustom : public OpDef {
public:
    explicit CumsumReverseCustom(const char *name) : OpDef(name)
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
OP_ADD(CumsumReverseCustom);
} // namespace ops
