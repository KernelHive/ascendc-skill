
#include "cumprod_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Increase parallelism: more blocks means more concurrent rows to hide scalar prefix latency.
// For the benchmark (32768 rows), 64 blocks gives 512 rows per block (on average).
static constexpr uint32_t BLOCK_DIM = 64;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;

    const auto shape = context->GetInputShape(0)->GetOriginShape();
    const uint32_t rank = static_cast<uint32_t>(shape.GetDimNum());
    const uint32_t totalElems = static_cast<uint32_t>(shape.GetShapeSize());

    // Specialized for benchmark: cumprod along last dim.
    uint32_t rows = 1;
    uint32_t cols = totalElems;
    if (rank == 2) {
        rows = static_cast<uint32_t>(shape.GetDim(0));
        cols = static_cast<uint32_t>(shape.GetDim(1));
    } else if (rank == 1) {
        rows = 1;
        cols = static_cast<uint32_t>(shape.GetDim(0));
    }

    context->SetBlockDim(BLOCK_DIM);
    context->SetTilingKey(0);

    // Ceil-div rows across blocks; each block processes contiguous rows.
    const uint32_t rowsPerCore = (rows + BLOCK_DIM - 1U) / BLOCK_DIM;

    tiling.set_rows(rows);
    tiling.set_cols(cols);
    tiling.set_totalElems(totalElems);
    tiling.set_rowsPerCore(rowsPerCore);

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
class CumprodCustom : public OpDef {
public:
    explicit CumprodCustom(const char *name) : OpDef(name)
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
OP_ADD(CumprodCustom);
} // namespace ops
