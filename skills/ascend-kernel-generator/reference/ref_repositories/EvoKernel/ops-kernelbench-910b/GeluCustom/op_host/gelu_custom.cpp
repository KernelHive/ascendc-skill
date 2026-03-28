
#include "gelu_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

// Raise concurrency safely; keep moderate to avoid runtime instability.
static constexpr uint32_t BLOCK_DIM = 24;

// Larger tile to amortize queue/scalar overhead. UB usage:
// in(2)+out(2)=4*tile*4 = 4*8192*4=128KB plus tanh tmp (~tile*2..4) => still safe on 910B.
static constexpr uint32_t TILE_LENGTH = 8192;

static inline uint32_t AlignUp(uint32_t x, uint32_t a) { return (x + a - 1u) / a * a; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GeluCustomTilingData tiling;
    const uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(TILE_LENGTH);

    // Tanh scratch: proportional and aligned to reduce UB bank-group conflicts.
    // Empirically safe heuristic: 4 bytes/elt, aligned to 2KB.
    uint32_t tanhTmpBytes = TILE_LENGTH * 4u;
    tanhTmpBytes = AlignUp(tanhTmpBytes, 2048u);
    // Keep a small absolute floor for very small tiles.
    if (tanhTmpBytes < 4096u) tanhTmpBytes = 4096u;
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

class GeluCustom : public OpDef {
public:
    explicit GeluCustom(const char* name) : OpDef(name)
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

OP_ADD(GeluCustom);

}  // namespace ops
