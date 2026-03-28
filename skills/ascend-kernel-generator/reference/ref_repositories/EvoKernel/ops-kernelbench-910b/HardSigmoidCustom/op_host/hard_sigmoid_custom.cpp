
#include "hard_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t x, uint32_t y) { return (x + y - 1u) / y; }
static inline uint32_t AlignDownU32(uint32_t x, uint32_t a) { return (x / a) * a; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    HardSigmoidCustomTilingData tiling;

    const uint32_t totalLength =
        static_cast<uint32_t>(context->GetInputShape(0)->GetOriginShape().GetShapeSize());

    // Elementwise op: keep block count moderate to avoid device/runtime pressure for tiny inputs.
    uint32_t blockDim = 48;
    if (totalLength < (1u << 20)) { // < ~1M elems
        blockDim = 24;
    }
    const uint32_t minTileWork = 4096;
    blockDim = std::max(1u, std::min(blockDim, CeilDivU32(totalLength == 0 ? 1u : totalLength, minTileWork)));
    context->SetBlockDim(blockDim);

    // UB double-buffered in/out: ~16 * tile bytes for fp32 (2 bufs * (in+out) * 4B).
    uint32_t tile = 8192;
    if (totalLength < 8192u) tile = 4096;
    tile = AlignDownU32(tile, 256);
    if (tile == 0) tile = 256;

    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(tile);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
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
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class HardSigmoidCustom : public OpDef {
public:
    explicit HardSigmoidCustom(const char* name) : OpDef(name)
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

OP_ADD(HardSigmoidCustom);

}  // namespace ops
