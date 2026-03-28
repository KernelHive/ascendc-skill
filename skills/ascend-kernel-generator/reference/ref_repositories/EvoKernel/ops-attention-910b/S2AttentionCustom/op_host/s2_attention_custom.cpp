
#include "s2_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {
static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    S2AttentionCustomTilingData tiling;

    auto attn_shape = context->GetInputShape(0)->GetStorageShape(); // [B,3,C]
    auto x_shape    = context->GetInputShape(1)->GetStorageShape(); // [B,3,H,W,C]

    if (attn_shape.GetDimNum() != 3 || x_shape.GetDimNum() != 5) return ge::GRAPH_FAILED;

    const uint32_t B = static_cast<uint32_t>(attn_shape.GetDim(0));
    const uint32_t K = static_cast<uint32_t>(attn_shape.GetDim(1));
    const uint32_t C = static_cast<uint32_t>(attn_shape.GetDim(2));

    const uint32_t xB = static_cast<uint32_t>(x_shape.GetDim(0));
    const uint32_t xK = static_cast<uint32_t>(x_shape.GetDim(1));
    const uint32_t H  = static_cast<uint32_t>(x_shape.GetDim(2));
    const uint32_t W  = static_cast<uint32_t>(x_shape.GetDim(3));
    const uint32_t xC = static_cast<uint32_t>(x_shape.GetDim(4));

    if (B == 0 || C == 0 || H == 0 || W == 0) return ge::GRAPH_FAILED;
    if (K != 3) return ge::GRAPH_FAILED;
    if (xB != B || xK != K || xC != C) return ge::GRAPH_FAILED;

    const uint64_t HW64 = (uint64_t)H * (uint64_t)W;
    const uint64_t rows64 = (uint64_t)B * HW64;
    if (HW64 > 0xFFFFFFFFULL || rows64 > 0xFFFFFFFFULL) return ge::GRAPH_FAILED;
    const uint32_t HW   = (uint32_t)HW64;
    const uint32_t rows = (uint32_t)rows64;

    // Choose a C tile that is vector-friendly and fits UB easily.
    // 32 floats = 128B alignment helps MTE and vector ops; cap at 256 for less loop overhead.
    uint32_t cTile = 256U;
    if (C < cTile) cTile = 32U * CeilDivU32(C, 32U);
    cTile = std::max<uint32_t>(32U, std::min<uint32_t>(cTile, 256U));
    cTile = 32U * CeilDivU32(cTile, 32U);

    // Parallelize over rows (b,hw). Each row does C work in tiles.
    uint32_t idealCores = 48U;
    uint32_t coreNum = std::min<uint32_t>(idealCores, std::max<uint32_t>(1U, rows));
    context->SetBlockDim(coreNum);

    uint32_t rowsPerCore = CeilDivU32(rows, coreNum);

    tiling.set_B(B);
    tiling.set_K(K);
    tiling.set_H(H);
    tiling.set_W(W);
    tiling.set_C(C);
    tiling.set_HW(HW);
    tiling.set_rows(rows);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_cTile(cTile);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class S2AttentionCustom : public OpDef {
public:
    explicit S2AttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("attn")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x_all")
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
OP_ADD(S2AttentionCustom);
} // namespace ops
