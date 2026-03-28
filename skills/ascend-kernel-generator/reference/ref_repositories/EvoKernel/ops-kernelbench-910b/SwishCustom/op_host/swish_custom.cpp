
#include "swish_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

// Adaptive blockDim: high enough to hide vector/scalar latency but avoid oversubscription.
static constexpr uint32_t kMaxBlockDim = 32;
static constexpr uint32_t kMinBlockDim = 8;

// Keep 8192 to avoid the known regression from smaller tiles; good MTE burst/UB usage balance.
static constexpr uint32_t kTileSize = 8192;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SwishCustomTilingData tiling;
    const uint32_t totalLength =
        static_cast<uint32_t>(context->GetInputShape(0)->GetOriginShape().GetShapeSize());

    // Rough heuristic: 1 block per ~4M elems, clamped.
    uint32_t blockDim = totalLength / (4u * 1024u * 1024u);
    blockDim = std::max(kMinBlockDim, std::min(kMaxBlockDim, blockDim));
    if (totalLength < blockDim) blockDim = 1;
    context->SetBlockDim(blockDim);

    tiling.set_totalLength(totalLength);
    tiling.set_blockDim(blockDim);
    tiling.set_tileSize(kTileSize);

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

class SwishCustom : public OpDef {
public:
    explicit SwishCustom(const char* name) : OpDef(name)
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

OP_ADD(SwishCustom);

}  // namespace ops
