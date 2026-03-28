
#include "softmax_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static constexpr uint32_t kMaxBlockDim = 24;

// UB-aware tile selection (elements). Keep aligned for better MTE efficiency.
static constexpr uint32_t kAlign = 256;
static constexpr uint32_t kDefaultTile = 8192;
static constexpr uint32_t kMinTile = 2048;
static constexpr uint32_t kMaxTile = 8192;

// VECCALC is now only reduction workspace + scalar (no tile-sized vec), so it's very safe.
static constexpr uint32_t kReduceWorkFloats = 4096;
static constexpr uint32_t kScalarFloats = 256;
static constexpr uint32_t kMaxCalcFloats = 65536;

static inline uint32_t AlignDown(uint32_t v, uint32_t a) {
    return (a == 0) ? v : (v / a) * a;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SoftmaxCustomTilingData tiling;

    const auto* inShape = context->GetInputShape(0);
    const gert::Shape& shape = inShape->GetOriginShape();
    const int32_t dimNum = shape.GetDimNum();

    uint64_t cols64 = 1;
    uint64_t rows64 = 1;

    if (dimNum <= 0) {
        cols64 = 1;
        rows64 = 0;
    } else {
        cols64 = static_cast<uint64_t>(shape.GetDim(dimNum - 1));
        if (cols64 == 0) cols64 = 1;
        rows64 = 1;
        for (int32_t i = 0; i < dimNum - 1; ++i) {
            rows64 *= static_cast<uint64_t>(shape.GetDim(i));
        }
    }

    uint32_t cols = static_cast<uint32_t>(cols64);
    uint32_t rows = static_cast<uint32_t>(rows64);

    uint32_t blockDim = std::min<uint32_t>(kMaxBlockDim, std::max<uint32_t>(1u, rows == 0 ? 1u : rows));
    context->SetBlockDim(blockDim);

    uint32_t rowsPerCore = (rows + blockDim - 1) / blockDim;

    uint32_t tileCols = kDefaultTile;
    if (cols > 0 && tileCols > cols) tileCols = cols;
    tileCols = std::min<uint32_t>(tileCols, kMaxTile);
    tileCols = std::max<uint32_t>(tileCols, kMinTile);
    tileCols = AlignDown(tileCols, kAlign);
    if (tileCols == 0) tileCols = (cols > 0) ? std::min<uint32_t>(cols, kMinTile) : 1u;

    while (tileCols > 1 && (kReduceWorkFloats + kScalarFloats) > kMaxCalcFloats) {
        tileCols = AlignDown(tileCols / 2, kAlign);
        if (tileCols == 0) tileCols = 1;
    }
    if (cols > 0 && tileCols > cols) tileCols = cols;
    if (cols == 0) tileCols = 1;
    if (tileCols == 0) tileCols = 1;

    tiling.set_rows(rows);
    tiling.set_cols(cols);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_tileCols(tileCols);

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
class SoftmaxCustom : public OpDef {
public:
    explicit SoftmaxCustom(const char* name) : OpDef(name)
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

OP_ADD(SoftmaxCustom);
}  // namespace ops
