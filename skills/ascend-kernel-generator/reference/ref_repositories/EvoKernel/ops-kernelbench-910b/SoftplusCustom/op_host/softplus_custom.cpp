
#include "softplus_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static constexpr uint32_t kMaxBlockDim = 32;
// Large tile to reduce MTE transaction count; will be aligned to 256B (64 fp32).
static constexpr uint32_t kTileSize = 8192;

static inline uint32_t AlignUp(uint32_t x, uint32_t a) {
    return (x + a - 1u) / a * a;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SoftplusCustomTilingData tiling;

    const uint32_t totalLength =
        static_cast<uint32_t>(context->GetInputShape(0)->GetOriginShape().GetShapeSize());

    uint32_t blockDim = 1;
    if (totalLength > 0) {
        const uint32_t tiles = (totalLength + kTileSize - 1) / kTileSize;
        // Keep enough blocks for latency hiding, but cap.
        blockDim = tiles < kMaxBlockDim ? tiles : kMaxBlockDim;
        if (blockDim == 0) blockDim = 1;
    }
    context->SetBlockDim(blockDim);

    uint32_t tileSize = AlignUp(kTileSize, 64);  // 256B alignment for fp32
    if (tileSize == 0) tileSize = 64;

    tiling.set_totalLength(totalLength);
    tiling.set_blockDim(blockDim);
    tiling.set_tileSize(tileSize);

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
class SoftplusCustom : public OpDef {
public:
    explicit SoftplusCustom(const char* name) : OpDef(name)
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

OP_ADD(SoftplusCustom);
}  // namespace ops
