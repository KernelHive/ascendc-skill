
#include "relu_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

// UB usage with double-buffered queues: in(2)+out(2)=4*tileLen*4=16*tileLen bytes.
// Pick a larger tile to reduce MTE setup overhead while staying safely within a UB slice.
static constexpr uint32_t UB_BUDGET_BYTES = 192 * 1024;   // conservative slice
static constexpr uint32_t ALIGN_ELEMS     = 2048;         // favor larger bursts than baseline
static constexpr uint32_t HARD_CAP_ELEMS  = 24576;        // keep latency per iter reasonable

static inline uint32_t AlignDown(uint32_t x, uint32_t a) { return (x / a) * a; }
static inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ReluCustomTilingData tiling;
    const uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    // Compute max tile length from UB budget (elements).
    uint32_t maxTile = UB_BUDGET_BYTES / 16;  // elements
    maxTile = AlignDown(maxTile, ALIGN_ELEMS);
    if (maxTile < ALIGN_ELEMS) maxTile = ALIGN_ELEMS;
    if (maxTile > HARD_CAP_ELEMS) maxTile = AlignDown(HARD_CAP_ELEMS, ALIGN_ELEMS);

    uint32_t tileLen = maxTile;
    if (totalLength < tileLen) {
        // For very small tensors, allow non-aligned tail tileLen.
        uint32_t aligned = AlignDown(totalLength, ALIGN_ELEMS);
        tileLen = (aligned > 0) ? aligned : totalLength;
        if (tileLen == 0) tileLen = totalLength;
    }

    const uint32_t numTiles = (tileLen == 0) ? 0 : CeilDiv(totalLength, tileLen);
    // Favor enough parallelism to hide memory latency but avoid too many tiny blocks.
    uint32_t blockDim = numTiles;
    if (blockDim < 8u) blockDim = 8u;
    if (blockDim > 32u) blockDim = 32u;  // slightly higher ceiling than baseline
    context->SetBlockDim(blockDim);

    // Precompute full-tile and tail flags for a steady-state kernel loop.
    const uint32_t fullTiles = (tileLen == 0) ? 0 : (totalLength / tileLen);
    const uint32_t hasTail = (tileLen == 0) ? 0 : ((totalLength % tileLen) != 0 ? 1u : 0u);

    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(tileLen);
    tiling.set_blockDim(blockDim);
    tiling.set_fullTilesPerBlock(fullTiles);
    tiling.set_hasTail(hasTail);

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

class ReluCustom : public OpDef {
public:
    explicit ReluCustom(const char* name) : OpDef(name)
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

OP_ADD(ReluCustom);

}  // namespace ops
