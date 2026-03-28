
#include "min_gpt_new_gelu_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Compute-heavy pointwise (tanh + several mul/add). Favor more blocks for concurrency.
static constexpr uint32_t BLOCK_DIM = 24;

// Big enough to amortize overhead; small enough to keep UB usage safe.
static constexpr uint32_t TILE_LENGTH = 8192;

static inline uint32_t AlignUp(uint32_t x, uint32_t a) { return (x + a - 1u) / a * a; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MinGptNewGeluCustomTilingData tiling;
    const uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    const uint32_t tiles = (totalLength + TILE_LENGTH - 1u) / TILE_LENGTH;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(TILE_LENGTH);
    tiling.set_tiles(tiles);

    // Avoid toolchain-dependent host helpers; use a compact, aligned heuristic.
    // Safe baseline for Tanh scratch: 2B/elem, clamped and aligned.
    uint32_t tanhTmpBytes = TILE_LENGTH * 2u;
    if (tanhTmpBytes < 8u * 1024u) tanhTmpBytes = 8u * 1024u;
    if (tanhTmpBytes > 64u * 1024u) tanhTmpBytes = 64u * 1024u;
    tanhTmpBytes = AlignUp(tanhTmpBytes, 256u);
    tiling.set_tanhTmpBytes(tanhTmpBytes);

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

class MinGptNewGeluCustom : public OpDef {
public:
    explicit MinGptNewGeluCustom(const char* name) : OpDef(name)
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

OP_ADD(MinGptNewGeluCustom);

}  // namespace ops
