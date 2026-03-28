
#include "stream_write_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static constexpr uint32_t BLOCK_DIM = 32;          // good occupancy on 910B
static constexpr uint32_t DEFAULT_TILE_C = 1024;   // vector-friendly

static inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
static inline uint32_t RoundUp(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    StreamWriteCustomTilingData tiling;

    const auto yShape = context->GetInputShape(0)->GetOriginShape(); // (B,T,C)
    const auto hShape = context->GetInputShape(1)->GetOriginShape(); // (B,T,N)

    const uint32_t B = static_cast<uint32_t>(yShape.GetDim(0));
    const uint32_t T = static_cast<uint32_t>(yShape.GetDim(1));
    const uint32_t C = static_cast<uint32_t>(yShape.GetDim(2));
    const uint32_t N = static_cast<uint32_t>(hShape.GetDim(2));
    const uint32_t BT = B * T;

    uint32_t tileC = DEFAULT_TILE_C;
    if (tileC > C) tileC = C;

    // Align tileC to 64 elements (256B for fp32) for better vector/MTE efficiency.
    tileC = (tileC / 64) * 64;
    if (tileC == 0) tileC = C;

    const uint32_t cTiles = CeilDiv(C, tileC);
    const uint32_t totalTiles = BT * cTiles;

    // Ensure h_post copy is >= 32B for fp32: Npad*4 >= 32 => Npad >= 8.
    const uint32_t Npad = RoundUp(N, 8);

    context->SetBlockDim(BLOCK_DIM);

    tiling.set_B(B);
    tiling.set_T(T);
    tiling.set_N(N);
    tiling.set_C(C);
    tiling.set_BT(BT);
    tiling.set_tileC(tileC);
    tiling.set_cTiles(cTiles);
    tiling.set_totalTiles(totalTiles);
    tiling.set_Npad(Npad);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* y_shape = context->GetInputShape(0);
    const gert::Shape* h_shape = context->GetInputShape(1);
    gert::Shape* out_shape = context->GetOutputShape(0);

    out_shape->SetDimNum(4);
    out_shape->SetDim(0, y_shape->GetDim(0)); // B
    out_shape->SetDim(1, y_shape->GetDim(1)); // T
    out_shape->SetDim(2, h_shape->GetDim(2)); // N
    out_shape->SetDim(3, y_shape->GetDim(2)); // C
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class StreamWriteCustom : public OpDef {
public:
    explicit StreamWriteCustom(const char* name) : OpDef(name)
    {
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("h_post")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(StreamWriteCustom);
} // namespace ops
