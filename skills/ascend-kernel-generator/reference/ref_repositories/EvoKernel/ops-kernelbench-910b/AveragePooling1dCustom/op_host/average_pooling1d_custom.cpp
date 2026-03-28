
#include "average_pooling1d_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AveragePooling1dCustomTilingData tiling;

    const auto inShape = context->GetInputShape(0)->GetStorageShape();
    if (inShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t n = static_cast<uint32_t>(inShape.GetDim(0));
    const uint32_t c = static_cast<uint32_t>(inShape.GetDim(1));
    const uint32_t l_in = static_cast<uint32_t>(inShape.GetDim(2));

    // Specialized parameters for this compiled operator:
    constexpr int32_t kernel_size = 8;
    constexpr int32_t stride = 1;
    constexpr int32_t padding = 4;

    const int64_t l_out64 =
        (static_cast<int64_t>(l_in) + 2LL * padding - static_cast<int64_t>(kernel_size)) / stride + 1LL;
    if (l_out64 <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t l_out = static_cast<uint32_t>(l_out64);

    const uint32_t totalX = static_cast<uint32_t>(inShape.GetShapeSize());
    const auto outShape = context->GetOutputShape(0)->GetStorageShape();
    const uint32_t totalY = static_cast<uint32_t>(outShape.GetShapeSize());

    tiling.set_n(n);
    tiling.set_c(c);
    tiling.set_l_in(l_in);
    tiling.set_l_out(l_out);
    tiling.set_totalX(totalX);
    tiling.set_totalY(totalY);
    tiling.set_nc(n * c);

    // Increase occupancy: parallelize over flattened output space.
    // Use a conservative cap to avoid excessive blocks; still large enough for this workload.
    uint32_t blockDim = 256;
    if (totalY < blockDim) {
        blockDim = std::max<uint32_t>(1, totalY);
    }
    context->SetBlockDim(blockDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class AveragePooling1dCustom : public OpDef {
public:
    explicit AveragePooling1dCustom(const char* name) : OpDef(name)
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

OP_ADD(AveragePooling1dCustom);

} // namespace ops
