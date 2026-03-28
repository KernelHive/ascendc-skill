
#include "triplet_margin_loss_custom_tiling.h"
#include "register/op_def_registry.h"
#include <string.h>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TripletMarginLossCustomTilingData tiling;

    auto shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t dimNum = static_cast<uint32_t>(shape.GetDimNum());

    uint32_t batchSize = 0;
    uint32_t featSize = 0;

    if (dimNum == 0) {
        batchSize = 1;
        featSize = 1;
    } else if (dimNum == 1) {
        batchSize = static_cast<uint32_t>(shape.GetDim(0));
        featSize = 1;
    } else {
        batchSize = static_cast<uint32_t>(shape.GetDim(0));
        featSize = 1;
        for (uint32_t i = 1; i < dimNum; ++i) {
            featSize *= static_cast<uint32_t>(shape.GetDim(i));
        }
    }

    // Single-block to keep deterministic mean reduction without atomics/workspace.
    context->SetBlockDim(1);

    constexpr uint32_t BLOCK_SIZE = 32;
    constexpr uint32_t TYPE_SIZE = 4;
    constexpr uint32_t ALIGN_NUM = BLOCK_SIZE / TYPE_SIZE; // 8 floats

    // Slightly larger tile to reduce loop overhead, still UB-safe with pingpong:
    // UB floats ~= 2*(3*tile) + 3*tile = 9*tile
    uint32_t featTile = 4096;

    if (featSize == 0) {
        featTile = ALIGN_NUM;
    } else if (featTile > featSize) {
        featTile = featSize;
    }
    if (featTile < ALIGN_NUM) featTile = ALIGN_NUM;
    featTile = ((featTile + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;

    uint32_t featTileNum = 1;
    uint32_t featLast = featSize;
    if (featSize != 0) {
        featTileNum = (featSize + featTile - 1) / featTile;
        if (featTileNum == 0) featTileNum = 1;
        featLast = featSize - (featTileNum - 1) * featTile;
        if (featLast == 0) featLast = featTile;
    } else {
        featTileNum = 1;
        featLast = 0;
    }

    float invBatch = 0.0f;
    if (batchSize != 0) {
        invBatch = 1.0f / static_cast<float>(batchSize);
    }

    tiling.set_batchSize(batchSize);
    tiling.set_featSize(featSize);
    tiling.set_featTile(featTile);
    tiling.set_featTileNum(featTileNum);
    tiling.set_featLast(featLast);
    tiling.set_invBatch(invBatch);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class TripletMarginLossCustom : public OpDef {
public:
    explicit TripletMarginLossCustom(const char* name) : OpDef(name)
    {
        this->Input("anchor")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("positive")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("negative")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("margin")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(TripletMarginLossCustom);

} // namespace ops
