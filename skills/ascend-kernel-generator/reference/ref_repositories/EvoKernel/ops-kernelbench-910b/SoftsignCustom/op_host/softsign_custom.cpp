
#include "softsign_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

// Use a fixed aligned tile to stabilize MTE transactions and UB access patterns.
static constexpr uint32_t kAlignElems = 256;
static constexpr uint32_t kTileSize = 8192;          // 8192 fp32 = 32KB per queue buffer
static constexpr uint32_t kMaxBlockDim = 48;
static constexpr uint32_t kTargetTilesPerCore = 4;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SoftsignCustomTilingData tiling;

    const auto* inShape = context->GetInputShape(0);
    const gert::Shape& shape = inShape->GetOriginShape();
    uint64_t total64 = shape.GetShapeSize();
    if (shape.GetDimNum() <= 0 && total64 == 0) total64 = 1;

    if (total64 > static_cast<uint64_t>(0xFFFFFFFFu)) {
        total64 = static_cast<uint64_t>(0xFFFFFFFFu);
    }
    const uint32_t total = static_cast<uint32_t>(total64);

    uint32_t tileSize = kTileSize;
    // keep alignment contract
    tileSize = (tileSize / kAlignElems) * kAlignElems;
    if (tileSize == 0) tileSize = kAlignElems;
    if (total > 0 && tileSize > total) tileSize = total;

    const uint32_t tiles = (total + tileSize - 1u) / tileSize;

    uint32_t blockDim = 1;
    if (tiles > 0) {
        blockDim = (tiles + kTargetTilesPerCore - 1u) / kTargetTilesPerCore;
        if (blockDim == 0) blockDim = 1;
        blockDim = std::min<uint32_t>(blockDim, kMaxBlockDim);
    }
    context->SetBlockDim(blockDim);

    tiling.set_totalLength(total);
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
class SoftsignCustom : public OpDef {
public:
    explicit SoftsignCustom(const char* name) : OpDef(name)
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

OP_ADD(SoftsignCustom);
}  // namespace ops
