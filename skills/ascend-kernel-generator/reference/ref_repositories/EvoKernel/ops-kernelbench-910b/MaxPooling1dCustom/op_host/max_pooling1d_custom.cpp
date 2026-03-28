
#include "max_pooling1d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MaxPooling1dCustomTilingData tiling;

    const auto inShape = context->GetInputShape(0)->GetStorageShape();
    if (inShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t n = static_cast<uint32_t>(inShape.GetDim(0));
    const uint32_t c = static_cast<uint32_t>(inShape.GetDim(1));
    const uint32_t l_in = static_cast<uint32_t>(inShape.GetDim(2));

    constexpr int32_t kernel_size = 8;
    constexpr int32_t stride = 1;
    constexpr int32_t padding = 4;
    constexpr int32_t dilation = 1;

    const int64_t l_out64 =
        (static_cast<int64_t>(l_in) + 2LL * padding
         - (static_cast<int64_t>(kernel_size) - 1LL) * dilation - 1LL) / stride + 1LL;
    if (l_out64 <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t l_out = static_cast<uint32_t>(l_out64);

    const uint64_t totalRows64 = static_cast<uint64_t>(n) * c;
    if (totalRows64 == 0 || totalRows64 > 0xFFFFFFFFULL) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t totalRows = static_cast<uint32_t>(totalRows64);

    // Increase parallelism safely; cap blocks to avoid oversubscription/launch overhead.
    // This kernel is scalar/control heavy; more blocks help hide pipeline gaps.
    constexpr uint32_t kMaxBlocks = 192;
    uint32_t blockDim = totalRows;
    if (blockDim > kMaxBlocks) blockDim = kMaxBlocks;
    if (blockDim < 1) blockDim = 1;

    const uint32_t rowsPerBlock = (totalRows + blockDim - 1) / blockDim;

    tiling.set_n(n);
    tiling.set_c(c);
    tiling.set_l_in(l_in);
    tiling.set_l_out(l_out);
    tiling.set_totalRows(totalRows);
    tiling.set_blockDim(blockDim);
    tiling.set_rowsPerBlock(rowsPerBlock);

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

class MaxPooling1dCustom : public OpDef {
public:
    explicit MaxPooling1dCustom(const char* name) : OpDef(name)
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

OP_ADD(MaxPooling1dCustom);

} // namespace ops
