
#include "average_pooling3d_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <algorithm>

namespace optiling {

static inline int64_t OutSize1D(int64_t in, int64_t k, int64_t s, int64_t p)
{
    return (in + 2 * p - k) / s + 1;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AveragePooling3dCustomTilingData tiling;

    const auto inShape = context->GetInputShape(0)->GetStorageShape();
    if (inShape.GetDimNum() != 5) return ge::GRAPH_FAILED;

    const uint32_t n    = static_cast<uint32_t>(inShape.GetDim(0));
    const uint32_t c    = static_cast<uint32_t>(inShape.GetDim(1));
    const uint32_t d_in = static_cast<uint32_t>(inShape.GetDim(2));
    const uint32_t h_in = static_cast<uint32_t>(inShape.GetDim(3));
    const uint32_t w_in = static_cast<uint32_t>(inShape.GetDim(4));

    constexpr int64_t K = 3;
    constexpr int64_t S = 2;
    constexpr int64_t P = 1;

    const int64_t d_out64 = OutSize1D(static_cast<int64_t>(d_in), K, S, P);
    const int64_t h_out64 = OutSize1D(static_cast<int64_t>(h_in), K, S, P);
    const int64_t w_out64 = OutSize1D(static_cast<int64_t>(w_in), K, S, P);
    if (d_out64 <= 0 || h_out64 <= 0 || w_out64 <= 0) return ge::GRAPH_FAILED;

    const uint32_t d_out = static_cast<uint32_t>(d_out64);
    const uint32_t h_out = static_cast<uint32_t>(h_out64);
    const uint32_t w_out = static_cast<uint32_t>(w_out64);

    const uint64_t rows64 = static_cast<uint64_t>(n) * static_cast<uint64_t>(c) *
                            static_cast<uint64_t>(d_out) * static_cast<uint64_t>(h_out);
    if (rows64 == 0 || rows64 > 0xFFFFFFFFull) return ge::GRAPH_FAILED;
    const uint32_t rows = static_cast<uint32_t>(rows64);

    // More conservative cap than previous attempt:
    // keep enough blocks for occupancy, but increase rowsPerBlock to amortize scalar setup.
    uint32_t blockDim = 1;
    if (rows >= 128U * 256U) blockDim = 128;
    else if (rows >= 64U * 256U)  blockDim = 64;
    else if (rows >= 32U * 256U)  blockDim = 32;
    else if (rows >= 16U * 256U)  blockDim = 16;
    else if (rows >= 8U * 256U)   blockDim = 8;
    else if (rows >= 4U * 256U)   blockDim = 4;
    else if (rows >= 2U * 256U)   blockDim = 2;

    blockDim = std::min<uint32_t>(blockDim, rows);
    blockDim = std::max<uint32_t>(blockDim, 1u);
    context->SetBlockDim(blockDim);

    const uint32_t rowsPerBlock = (rows + blockDim - 1U) / blockDim;

    tiling.set_n(n);
    tiling.set_c(c);
    tiling.set_d_in(d_in);
    tiling.set_h_in(h_in);
    tiling.set_w_in(w_in);
    tiling.set_d_out(d_out);
    tiling.set_h_out(h_out);
    tiling.set_w_out(w_out);
    tiling.set_rows(rows);
    tiling.set_blockDim(blockDim);
    tiling.set_rowsPerBlock(rowsPerBlock);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class AveragePooling3dCustom : public OpDef {
public:
    explicit AveragePooling3dCustom(const char* name) : OpDef(name)
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

OP_ADD(AveragePooling3dCustom);

} // namespace ops
