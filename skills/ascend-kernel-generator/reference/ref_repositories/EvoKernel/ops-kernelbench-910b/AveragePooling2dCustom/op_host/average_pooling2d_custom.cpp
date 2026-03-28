
#include "average_pooling2d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AveragePooling2dCustomTilingData tiling;

    const auto inShape = context->GetInputShape(0)->GetStorageShape();
    if (inShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t n = static_cast<uint32_t>(inShape.GetDim(0));
    const uint32_t c = static_cast<uint32_t>(inShape.GetDim(1));
    const uint32_t h_in = static_cast<uint32_t>(inShape.GetDim(2));
    const uint32_t w_in = static_cast<uint32_t>(inShape.GetDim(3));

    // Specialized params for this build: kernel=11, stride=11, padding=0, ceil_mode=False
    constexpr int32_t KH = 11;
    constexpr int32_t KW = 11;
    constexpr int32_t SH = 11;
    constexpr int32_t SW = 11;
    constexpr int32_t PH = 0;
    constexpr int32_t PW = 0;

    const int64_t h_out64 =
        (static_cast<int64_t>(h_in) + 2LL * PH - static_cast<int64_t>(KH)) / SH + 1LL;
    const int64_t w_out64 =
        (static_cast<int64_t>(w_in) + 2LL * PW - static_cast<int64_t>(KW)) / SW + 1LL;

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

    // Keep the same safe occupancy approach, but a bit smoother for large outputs.
    // Avoid extreme block counts to reduce risk of runtime instability.
    uint32_t blockDim = 1;
    if (totalY >= 128u * 4096u) blockDim = 96;
    else if (totalY >= 64u * 4096u) blockDim = 64;
    else if (totalY >= 32u * 4096u) blockDim = 32;
    else if (totalY >= 16u * 4096u) blockDim = 16;
    else if (totalY >= 8u * 4096u)  blockDim = 8;
    else if (totalY >= 4u * 4096u)  blockDim = 4;
    else if (totalY >= 2u * 4096u)  blockDim = 2;
    context->SetBlockDim(blockDim);

    // Linear chunk per block (ceil-div), then align to micro-batch=4 outputs for kernel ILP.
    uint32_t elemsPerBlock = (totalY + blockDim - 1) / blockDim;
    constexpr uint32_t MICRO = 4;
    elemsPerBlock = (elemsPerBlock + MICRO - 1) / MICRO * MICRO;
    if (elemsPerBlock == 0) elemsPerBlock = MICRO;

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

class AveragePooling2dCustom : public OpDef {
public:
    explicit AveragePooling2dCustom(const char* name) : OpDef(name)
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

OP_ADD(AveragePooling2dCustom);

} // namespace ops
