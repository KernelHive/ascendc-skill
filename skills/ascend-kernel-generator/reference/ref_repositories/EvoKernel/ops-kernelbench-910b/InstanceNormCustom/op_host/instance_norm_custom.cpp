
#include "instance_norm_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <algorithm>

namespace optiling {

static inline uint32_t AlignUp(uint32_t x, uint32_t a) {
    return (x + a - 1U) / a * a;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    InstanceNormCustomTilingData tiling;

    constexpr float EPS = 1e-5f;

    const auto* inShape = context->GetInputShape(0);
    const auto& s = inShape->GetOriginShape();
    const size_t rank = s.GetDimNum();
    if (rank != 4) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t n64 = static_cast<uint64_t>(s.GetDim(0));
    const uint64_t c64 = static_cast<uint64_t>(s.GetDim(1));
    const uint64_t h64 = static_cast<uint64_t>(s.GetDim(2));
    const uint64_t w64 = static_cast<uint64_t>(s.GetDim(3));

    if (n64 == 0 || c64 == 0 || h64 == 0 || w64 == 0) {
        return ge::GRAPH_FAILED;
    }
    if (n64 > 0xFFFFFFFFULL || c64 > 0xFFFFFFFFULL || h64 > 0xFFFFFFFFULL || w64 > 0xFFFFFFFFULL) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t hw64 = h64 * w64;
    if (hw64 == 0 || hw64 > 0xFFFFFFFFULL) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t planes64 = n64 * c64;
    if (planes64 == 0 || planes64 > 0xFFFFFFFFULL) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t n = static_cast<uint32_t>(n64);
    const uint32_t c = static_cast<uint32_t>(c64);
    const uint32_t hw = static_cast<uint32_t>(hw64);
    const uint32_t planes = static_cast<uint32_t>(planes64);

    // Use multiple blocks safely (each block handles disjoint planes).
    // Cap to a small number to reduce overhead; tuning point.
    uint32_t blockDim = std::min<uint32_t>(planes, 32U);
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    // Tile length in elements for HxW streaming.
    // Needs to fit x tile, temp tile, y tile, plus reduce tmp and a few scalars.
    // Keep conservative for robustness.
    uint32_t tileLength = 4096U;
    if (tileLength > hw) tileLength = hw;
    if (tileLength == 0) tileLength = 1;

    // ReduceSum shared tmp; keep 32-float alignment and >= tileLength.
    const uint32_t reduceTmpLen = AlignUp(tileLength, 32U);

    tiling.set_n(n);
    tiling.set_c(c);
    tiling.set_hw(hw);
    tiling.set_planes(planes);

    tiling.set_blockDim(blockDim);
    tiling.set_tileLength(tileLength);
    tiling.set_reduceTmpLen(reduceTmpLen);

    tiling.set_invHw(1.0f / static_cast<float>(hw));
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

class InstanceNormCustom : public OpDef {
public:
    explicit InstanceNormCustom(const char* name) : OpDef(name)
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

OP_ADD(InstanceNormCustom);

}  // namespace ops
