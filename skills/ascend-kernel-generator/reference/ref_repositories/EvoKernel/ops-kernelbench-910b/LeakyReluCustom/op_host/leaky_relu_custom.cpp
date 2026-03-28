
#include "leaky_relu_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

// UB model for this kernel:
//   inQueueX: 2 * tile
//   outQueueY: 2 * tile
//   tmpQueue: 1 * tile
// Total floats = 5 * tile => bytes = 20 * tile
static constexpr uint32_t UB_BUDGET_BYTES = 192 * 1024;  // conservative per-core share
static constexpr uint32_t ALIGN_ELEMS = 2048;            // favor larger DMA bursts
static constexpr uint32_t HARD_CAP_ELEMS = 32768;        // avoid too-large per-iter latency

static inline uint32_t AlignDown(uint32_t x, uint32_t a)
{
    return (x / a) * a;
}

static inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1u) / b;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    LeakyReluCustomTilingData tiling;
    const uint32_t totalLength =
        static_cast<uint32_t>(context->GetInputShape(0)->GetOriginShape().GetShapeSize());

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *negativeSlope = attrs->GetAttrPointer<float>(0);

    // tileLen from UB budget: bytes = 20 * tileLen
    uint32_t maxTile = UB_BUDGET_BYTES / 20;  // elements
    maxTile = AlignDown(maxTile, ALIGN_ELEMS);
    if (maxTile < ALIGN_ELEMS) maxTile = ALIGN_ELEMS;
    if (maxTile > HARD_CAP_ELEMS) maxTile = AlignDown(HARD_CAP_ELEMS, ALIGN_ELEMS);

    uint32_t tileLen = maxTile;
    if (totalLength < tileLen) {
        const uint32_t aligned = AlignDown(totalLength, ALIGN_ELEMS);
        tileLen = (aligned > 0) ? aligned : totalLength;
    }

    // Adaptive blockDim based on number of tiles; keep conservative for stability.
    const uint32_t numTiles = (tileLen == 0) ? 1u : CeilDiv(totalLength, tileLen);
    uint32_t blockDim = numTiles;
    if (blockDim < 8u) blockDim = 8u;
    if (blockDim > 24u) blockDim = 24u;

    context->SetBlockDim(blockDim);

    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(tileLen);
    tiling.set_blockDim(blockDim);
    tiling.set_negativeSlope(*negativeSlope);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x_dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class LeakyReluCustom : public OpDef {
public:
    explicit LeakyReluCustom(const char* name) : OpDef(name)
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

        this->Attr("negative_slope").AttrType(OPTIONAL).Float(0.01);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LeakyReluCustom);
} // namespace ops
