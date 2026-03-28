
#include "l2_norm_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Keep stable parallelism; focus is per-core efficiency.
static constexpr uint32_t BLOCK_DIM = 8;

// Larger tile amortizes DMA/setup; we removed per-tile padding so this is safer.
// 6144 floats = 24KB; kernel uses 2 UB buffers => ~48KB, typically safe on 910B.
static constexpr uint32_t TILE_LENGTH = 6144;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    L2NormCustomTilingData tiling;
    constexpr float EPS = 1.0e-12f;

    const auto* inShape = context->GetInputShape(0);
    const auto& s = inShape->GetOriginShape();
    const size_t rank = s.GetDimNum();

    uint64_t cols64 = 0;
    uint64_t rows64 = 0;

    if (rank >= 2) {
        cols64 = static_cast<uint64_t>(s.GetDim(1));
        uint64_t left = static_cast<uint64_t>(s.GetDim(0));
        uint64_t right = 1;
        for (size_t i = 2; i < rank; ++i) {
            right *= static_cast<uint64_t>(s.GetDim(i));
        }
        rows64 = left * right;
    }

    if (cols64 > 0xFFFFFFFFULL) cols64 = 0xFFFFFFFFULL;
    if (rows64 > 0xFFFFFFFFULL) rows64 = 0xFFFFFFFFULL;
    const uint32_t cols = static_cast<uint32_t>(cols64);
    const uint32_t rows = static_cast<uint32_t>(rows64);

    uint32_t blockDim = BLOCK_DIM;
    if (rows > 0 && rows < blockDim) blockDim = rows;
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    tiling.set_rows(rows);
    tiling.set_cols(cols);
    tiling.set_tileLength(TILE_LENGTH);
    tiling.set_eps(EPS);

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

class L2NormCustom : public OpDef {
public:
    explicit L2NormCustom(const char* name) : OpDef(name)
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

OP_ADD(L2NormCustom);

}  // namespace ops
