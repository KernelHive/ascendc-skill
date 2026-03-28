
#include "deep_narrow_mlp_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <algorithm>

namespace optiling {

// Specialized benchmark contract (fixed shapes for performance & predictability)
static constexpr uint32_t BATCH = 1024;
static constexpr uint32_t IN = 8192;
static constexpr uint32_t HIDDEN = 1024;
static constexpr uint32_t NUM_HIDDEN = 16;
static constexpr uint32_t OUT = 8192;

static inline uint64_t ExpectedWElems()
{
    // W0: [HIDDEN, IN]
    // W1..W15: 15 blocks [HIDDEN, HIDDEN]
    // Wfinal: [OUT, HIDDEN]
    return (uint64_t)HIDDEN * (uint64_t)IN +
           (uint64_t)(NUM_HIDDEN - 1U) * (uint64_t)HIDDEN * (uint64_t)HIDDEN +
           (uint64_t)OUT * (uint64_t)HIDDEN;
}

static inline uint64_t ExpectedBElems()
{
    // b0..b15: 16 blocks [HIDDEN]
    // bfinal: [OUT]
    return (uint64_t)NUM_HIDDEN * (uint64_t)HIDDEN + (uint64_t)OUT;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto sx = context->GetInputShape(0);
    auto sw = context->GetInputShape(1);
    auto sb = context->GetInputShape(2);
    if (sx == nullptr || sw == nullptr || sb == nullptr) return ge::GRAPH_FAILED;

    const auto& x = sx->GetOriginShape();
    const auto& w = sw->GetOriginShape();
    const auto& b = sb->GetOriginShape();

    if (x.GetDimNum() != 2) return ge::GRAPH_FAILED;
    if (w.GetDimNum() != 1) return ge::GRAPH_FAILED;
    if (b.GetDimNum() != 1) return ge::GRAPH_FAILED;

    if ((uint32_t)x.GetDim(0) != BATCH || (uint32_t)x.GetDim(1) != IN) return ge::GRAPH_FAILED;

    // Packed tensors must match exact packed sizes.
    if ((uint64_t)w.GetDim(0) != ExpectedWElems()) return ge::GRAPH_FAILED;
    if ((uint64_t)b.GetDim(0) != ExpectedBElems()) return ge::GRAPH_FAILED;

    if (context->GetInputTensor(0)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(1)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(2)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;

    DeepNarrowMlpCustomTilingData tiling;
    tiling.set_totalX(sx->GetStorageShape().GetShapeSize());
    tiling.set_totalW(sw->GetStorageShape().GetShapeSize());
    tiling.set_totalB(sb->GetStorageShape().GetShapeSize());
    tiling.set_totalY(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());

    tiling.set_batch(BATCH);
    tiling.set_inSize(IN);
    tiling.set_hiddenSize(HIDDEN);
    tiling.set_outSize(OUT);
    tiling.set_numHidden(NUM_HIDDEN);

    // Conservative blockDim to reduce runtime pressure; stable mapping.
    uint32_t blockDim = 16;
    if (blockDim > BATCH) blockDim = BATCH;
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    uint32_t rowsPerBlock = (BATCH + blockDim - 1) / blockDim;
    if (rowsPerBlock == 0) rowsPerBlock = 1;
    tiling.set_rowsPerBlock(rowsPerBlock);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class DeepNarrowMlpCustom : public OpDef {
public:
    explicit DeepNarrowMlpCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("w_packed")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("b_packed")
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

OP_ADD(DeepNarrowMlpCustom);

} // namespace ops
