
#include "max_pooling3d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static inline int64_t OutDimFloor(int64_t in)
{
    // floor((in + 2P - (K-1)*D - 1)/S + 1) for K=3,S=2,P=1,D=1
    constexpr int64_t K = 3, S = 2, P = 1, D = 1;
    return (in + 2LL * P - (K - 1LL) * D - 1LL) / S + 1LL;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MaxPooling3dCustomTilingData tiling;

    const auto inShape = context->GetInputShape(0)->GetStorageShape();
    if (inShape.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t n    = static_cast<uint32_t>(inShape.GetDim(0));
    const uint32_t c    = static_cast<uint32_t>(inShape.GetDim(1));
    const uint32_t d_in = static_cast<uint32_t>(inShape.GetDim(2));
    const uint32_t h_in = static_cast<uint32_t>(inShape.GetDim(3));
    const uint32_t w_in = static_cast<uint32_t>(inShape.GetDim(4));

    const int64_t d_out64 = OutDimFloor(d_in);
    const int64_t h_out64 = OutDimFloor(h_in);
    const int64_t w_out64 = OutDimFloor(w_in);
    if (d_out64 <= 0 || h_out64 <= 0 || w_out64 <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t d_out = static_cast<uint32_t>(d_out64);
    const uint32_t h_out = static_cast<uint32_t>(h_out64);
    const uint32_t w_out = static_cast<uint32_t>(w_out64);

    const uint64_t totalRows64 = static_cast<uint64_t>(n) * c * d_out * h_out;
    if (totalRows64 == 0 || totalRows64 > 0xFFFFFFFFULL) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t totalRows = static_cast<uint32_t>(totalRows64);

    // Scalar-heavy kernel: raise block count to hide latency, but keep a safe cap.
    constexpr uint32_t kMaxBlocks = 192;
    uint32_t blockDim = totalRows;
    if (blockDim > kMaxBlocks) blockDim = kMaxBlocks;
    if (blockDim < 1) blockDim = 1;

    const uint32_t rowsPerBlock = (totalRows + blockDim - 1) / blockDim;

    tiling.set_n(n);
    tiling.set_c(c);
    tiling.set_d_in(d_in);
    tiling.set_h_in(h_in);
    tiling.set_w_in(w_in);
    tiling.set_d_out(d_out);
    tiling.set_h_out(h_out);
    tiling.set_w_out(w_out);
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

class MaxPooling3dCustom : public OpDef {
public:
    explicit MaxPooling3dCustom(const char* name) : OpDef(name)
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

OP_ADD(MaxPooling3dCustom);

} // namespace ops
