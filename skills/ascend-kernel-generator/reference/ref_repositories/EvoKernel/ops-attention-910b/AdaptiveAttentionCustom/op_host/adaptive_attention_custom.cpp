
#include "adaptive_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace optiling {

static inline uint32_t RoundDownPow2(uint32_t x) {
    if (x == 0) return 0;
    uint32_t p = 1;
    while ((p << 1) <= x) p <<= 1;
    return p;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    auto q_shape = context->GetInputShape(0)->GetStorageShape();
    auto k_shape = context->GetInputShape(1)->GetStorageShape();
    auto v_shape = context->GetInputShape(2)->GetStorageShape();

    if (q_shape.GetDimNum() != 4 || k_shape.GetDimNum() != 4 || v_shape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t B = static_cast<uint32_t>(q_shape.GetDim(0));
    const uint32_t H = static_cast<uint32_t>(q_shape.GetDim(1));
    const uint32_t S = static_cast<uint32_t>(q_shape.GetDim(2));
    const uint32_t D = static_cast<uint32_t>(q_shape.GetDim(3));
    if (B == 0 || H == 0 || S == 0 || D == 0) return ge::GRAPH_FAILED;

    for (int i = 0; i < 4; ++i) {
        if (static_cast<uint32_t>(k_shape.GetDim(i)) != static_cast<uint32_t>(q_shape.GetDim(i))) return ge::GRAPH_FAILED;
        if (static_cast<uint32_t>(v_shape.GetDim(i)) != static_cast<uint32_t>(q_shape.GetDim(i))) return ge::GRAPH_FAILED;
    }

    // Constraints kept to match the existing kernel limitations
    if (D > 128u) return ge::GRAPH_FAILED;
    if (S > 4096u) return ge::GRAPH_FAILED;

    AdaptiveAttentionCustomTilingData tiling;
    tiling.set_B(B);
    tiling.set_H(H);
    tiling.set_S(S);
    tiling.set_D(D);

    const uint64_t totalRows64 = (uint64_t)B * (uint64_t)H * (uint64_t)S;
    if (totalRows64 > 0xFFFFFFFFULL) return ge::GRAPH_FAILED;
    const uint32_t totalRows = (uint32_t)totalRows64;
    tiling.set_totalRows(totalRows);

    // Row-parallel kernel: use more cores to reduce serialization/pipeline gaps.
    // Cap at 48 (910B typical) but be safe if runtime uses smaller.
    const uint32_t coreNum = std::min<uint32_t>(std::max<uint32_t>(1u, totalRows), 48u);
    context->SetBlockDim(coreNum);
    tiling.set_coreNum(coreNum);

    // dTile: power-of-two <= D, capped at 64 (vector-friendly)
    uint32_t dTile = 64;
    if (D < 64) dTile = RoundDownPow2(D);
    if (dTile == 0) dTile = D;
    if (dTile > 64) dTile = 64;
    tiling.set_dTile(dTile);

    // sTile: keep moderate to fit UB; power-of-two for small S
    uint32_t sTile = 128;
    if (S < 128) sTile = RoundDownPow2(S);
    if (sTile == 0) sTile = S;
    if (sTile > 256) sTile = 256;
    // for large D reduce S tile to control UB
    if (D >= 128) sTile = std::min<uint32_t>(sTile, 64u);
    tiling.set_sTile(sTile);

    tiling.set_scale(1.0f / std::sqrt(static_cast<float>(D)));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
class AdaptiveAttentionCustom : public OpDef {
public:
    explicit AdaptiveAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v")
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
OP_ADD(AdaptiveAttentionCustom);
} // namespace ops
