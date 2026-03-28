
#include "grouped_query_attention_custom_tiling.h"
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

    // q: [B,H,S,D], k/v: [B,Hkv,S,D]
    auto q_shape = context->GetInputShape(0)->GetStorageShape();
    auto k_shape = context->GetInputShape(1)->GetStorageShape();
    auto v_shape = context->GetInputShape(2)->GetStorageShape();

    if (q_shape.GetDimNum() != 4 || k_shape.GetDimNum() != 4 || v_shape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t B   = static_cast<uint32_t>(q_shape.GetDim(0));
    const uint32_t H   = static_cast<uint32_t>(q_shape.GetDim(1));
    const uint32_t S   = static_cast<uint32_t>(q_shape.GetDim(2));
    const uint32_t D   = static_cast<uint32_t>(q_shape.GetDim(3));
    const uint32_t Hkv = static_cast<uint32_t>(k_shape.GetDim(1));

    if (B == 0 || H == 0 || S == 0 || D == 0 || Hkv == 0) return ge::GRAPH_FAILED;

    // Shape checks
    if (static_cast<uint32_t>(k_shape.GetDim(0)) != B || static_cast<uint32_t>(v_shape.GetDim(0)) != B) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(v_shape.GetDim(1)) != Hkv) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(k_shape.GetDim(2)) != S || static_cast<uint32_t>(v_shape.GetDim(2)) != S) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(k_shape.GetDim(3)) != D || static_cast<uint32_t>(v_shape.GetDim(3)) != D) return ge::GRAPH_FAILED;

    if (H % Hkv != 0) return ge::GRAPH_FAILED;
    const uint32_t G = H / Hkv;
    if (G == 0) return ge::GRAPH_FAILED;

    // Keep kernel constraints explicit (same spirit as successful MQA example).
    if (D > 128u) return ge::GRAPH_FAILED;
    if (S > 4096u) return ge::GRAPH_FAILED;

    GroupedQueryAttentionCustomTilingData tiling;
    tiling.set_B(B);
    tiling.set_H(H);
    tiling.set_Hkv(Hkv);
    tiling.set_G(G);
    tiling.set_S(S);
    tiling.set_D(D);
    tiling.set_totalBH(B * H);

    const uint32_t totalBH = B * H;
    const uint32_t coreNum = std::min<uint32_t>(std::max<uint32_t>(1u, totalBH), 24u);
    context->SetBlockDim(coreNum);
    tiling.set_coreNum(coreNum);

    uint32_t dTile = 128;
    if (D < 128) dTile = RoundDownPow2(D);
    if (dTile == 0) dTile = D;
    if (dTile > 128) dTile = 128;
    tiling.set_dTile(dTile);

    uint32_t sTile = 128;
    if (S < 128) sTile = RoundDownPow2(S);
    if (sTile == 0) sTile = S;
    if (sTile > 256) sTile = 256;
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
class GroupedQueryAttentionCustom : public OpDef {
public:
    explicit GroupedQueryAttentionCustom(const char* name) : OpDef(name)
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
OP_ADD(GroupedQueryAttentionCustom);
} // namespace ops
