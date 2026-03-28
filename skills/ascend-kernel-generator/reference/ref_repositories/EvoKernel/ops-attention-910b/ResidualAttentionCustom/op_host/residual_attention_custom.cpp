
#include "residual_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ResidualAttentionCustomTilingData tiling;

    const auto& xShape  = context->GetInputShape(0)->GetStorageShape();
    if (xShape.GetDimNum() != 4) return ge::GRAPH_FAILED;

    int64_t b64 = xShape.GetDim(0);
    int64_t c64 = xShape.GetDim(1);
    int64_t h64 = xShape.GetDim(2);
    int64_t w64 = xShape.GetDim(3);
    if (b64 <= 0 || c64 <= 0 || h64 <= 0 || w64 <= 0) return ge::GRAPH_FAILED;

    uint32_t B = static_cast<uint32_t>(b64);
    uint32_t C = static_cast<uint32_t>(c64);
    uint32_t H = static_cast<uint32_t>(h64);
    uint32_t W = static_cast<uint32_t>(w64);
    uint32_t HW = H * W;
    if (HW == 0) return ge::GRAPH_FAILED;

    uint64_t totalRows64 = static_cast<uint64_t>(B) * static_cast<uint64_t>(C);
    if (totalRows64 == 0 || totalRows64 > 0xFFFFFFFFULL) return ge::GRAPH_FAILED;
    uint32_t totalRows = static_cast<uint32_t>(totalRows64);

    tiling.set_B(B);
    tiling.set_C(C);
    tiling.set_H(H);
    tiling.set_W(W);
    tiling.set_HW(HW);
    tiling.set_totalRows(totalRows);
    tiling.set_invHW(1.0f / static_cast<float>(HW));

    // Increase parallelism: row reductions are independent and lightweight; more blocks helps hide scalar latency.
    uint32_t block_dim = std::min<uint32_t>(totalRows, 128);
    if (block_dim == 0) block_dim = 1;
    context->SetBlockDim(block_dim);

    uint32_t rowsPerCore = CeilDivU32(totalRows, block_dim);
    if (rowsPerCore == 0) rowsPerCore = 1;
    tiling.set_rowsPerCore(rowsPerCore);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
class ResidualAttentionCustom : public OpDef {
public:
    explicit ResidualAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("la")
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

OP_ADD(ResidualAttentionCustom);
} // namespace ops
