
#include "multi_query_attention_custom_tiling.h"
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

    // k/v must be [B,1,S,D]
    if (static_cast<uint32_t>(k_shape.GetDim(0)) != B || static_cast<uint32_t>(v_shape.GetDim(0)) != B) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(k_shape.GetDim(1)) != 1 || static_cast<uint32_t>(v_shape.GetDim(1)) != 1) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(k_shape.GetDim(2)) != S || static_cast<uint32_t>(v_shape.GetDim(2)) != S) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(k_shape.GetDim(3)) != D || static_cast<uint32_t>(v_shape.GetDim(3)) != D) return ge::GRAPH_FAILED;

    if (D > 128u) return ge::GRAPH_FAILED;
    if (S > 4096u) return ge::GRAPH_FAILED;

    MultiQueryAttentionCustomTilingData tiling;
    tiling.set_B(B);
    tiling.set_H(H);
    tiling.set_S(S);
    tiling.set_D(D);
    tiling.set_totalBH(B * H);

    // 2D parallelization: split over (BH) and query tiles.
    // total tasks = BH * ceil(S/qTile). Use up to 24 cores.
    uint32_t qTile = 8;
    if (S < 8) qTile = S;
    // Prefer power-of-2 tiles for simpler loops
    if (qTile != 0) qTile = RoundDownPow2(qTile);
    if (qTile == 0) qTile = 1;
    // Keep UB safe: qTile*D floats plus per-query scalars.
    // For typical D<=128, qTile=8 is safe and gives reuse.
    tiling.set_qTile(qTile);

    const uint32_t qGroups = (S + qTile - 1) / qTile;
    const uint32_t totalTasks = B * H * qGroups;
    const uint32_t coreNum = std::min<uint32_t>(std::max<uint32_t>(1u, totalTasks), 24u);
    context->SetBlockDim(coreNum);
    tiling.set_coreNum(coreNum);

    // sTile moderate to avoid UB bloat; ensure at least 64 if possible.
    uint32_t sTile = 128;
    if (S < 128) sTile = RoundDownPow2(S);
    if (sTile == 0) sTile = S;
    if (sTile < 64 && S >= 64) sTile = 64;
    if (sTile > 192) sTile = 192; // cap for UB safety with qTile buffers
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
class MultiQueryAttentionCustom : public OpDef {
public:
    explicit MultiQueryAttentionCustom(const char* name) : OpDef(name)
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
OP_ADD(MultiQueryAttentionCustom);
} // namespace ops
