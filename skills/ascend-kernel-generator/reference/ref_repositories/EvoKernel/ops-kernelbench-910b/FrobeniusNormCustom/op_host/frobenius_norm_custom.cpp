
#include "frobenius_norm_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Correct global Frobenius norm requires a global reduction; without atomics/cross-core sync,
// we run a single block and optimize its vector path.
static constexpr uint32_t BLOCK_DIM = 1;

// Tile size tuned for UB and vector ops (multiple of 256 bytes).
static constexpr uint32_t TILE_LENGTH = 8192;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    FrobeniusNormCustomTilingData tiling;

    const auto* inShape = context->GetInputShape(0);
    const auto& s = inShape->GetOriginShape();
    uint64_t total64 = s.GetShapeSize();
    if (total64 > 0xFFFFFFFFULL) total64 = 0xFFFFFFFFULL;
    const uint32_t totalLength = static_cast<uint32_t>(total64);

    context->SetBlockDim(BLOCK_DIM);

    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(TILE_LENGTH);

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

class FrobeniusNormCustom : public OpDef {
public:
    explicit FrobeniusNormCustom(const char* name) : OpDef(name)
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

OP_ADD(FrobeniusNormCustom);

}  // namespace ops
