
#include "max_pooling2d_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MaxPooling2dCustomTilingData tiling;

    const auto inShape = context->GetInputShape(0)->GetStorageShape();
    if (inShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t n = static_cast<uint32_t>(inShape.GetDim(0));
    const uint32_t c = static_cast<uint32_t>(inShape.GetDim(1));
    const uint32_t h_in = static_cast<uint32_t>(inShape.GetDim(2));
    const uint32_t w_in = static_cast<uint32_t>(inShape.GetDim(3));

    // Specialized params: kernel=4, stride=1, padding=1, dilation=1, ceil_mode=False
    constexpr int32_t K = 4;
    constexpr int32_t S = 1;
    constexpr int32_t P = 1;
    constexpr int32_t D = 1;

    const int64_t h_out64 =
        (static_cast<int64_t>(h_in) + 2LL * P - (static_cast<int64_t>(K) - 1LL) * D - 1LL) / S + 1LL;
    const int64_t w_out64 =
        (static_cast<int64_t>(w_in) + 2LL * P - (static_cast<int64_t>(K) - 1LL) * D - 1LL) / S + 1LL;

    if (h_out64 <= 0 || w_out64 <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t h_out = static_cast<uint32_t>(h_out64);
    const uint32_t w_out = static_cast<uint32_t>(w_out64);

    const uint64_t totalY64 = static_cast<uint64_t>(n) * static_cast<uint64_t>(c) *
                              static_cast<uint64_t>(h_out) * static_cast<uint64_t>(w_out);
    if (totalY64 == 0 || totalY64 > 0xFFFFFFFFULL) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t totalY = static_cast<uint32_t>(totalY64);

    // Stable, power-of-two-ish blockDim to improve occupancy without risky extremes.
    uint32_t blockDim = 1;
    if (totalY >= (1U << 20)) blockDim = 128;
    else if (totalY >= (1U << 19)) blockDim = 64;
    else if (totalY >= (1U << 18)) blockDim = 32;
    else if (totalY >= (1U << 17)) blockDim = 16;
    else if (totalY >= (1U << 16)) blockDim = 8;
    else if (totalY >= (1U << 15)) blockDim = 4;
    else if (totalY >= (1U << 14)) blockDim = 2;

    context->SetBlockDim(blockDim);
    const uint32_t elemsPerBlock = (totalY + blockDim - 1U) / blockDim;

    tiling.set_n(n);
    tiling.set_c(c);
    tiling.set_h_in(h_in);
    tiling.set_w_in(w_in);
    tiling.set_h_out(h_out);
    tiling.set_w_out(w_out);
    tiling.set_totalY(totalY);
    tiling.set_elemsPerBlock(elemsPerBlock);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class MaxPooling2dCustom : public OpDef {
public:
    explicit MaxPooling2dCustom(const char* name) : OpDef(name)
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

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MaxPooling2dCustom);

} // namespace ops
