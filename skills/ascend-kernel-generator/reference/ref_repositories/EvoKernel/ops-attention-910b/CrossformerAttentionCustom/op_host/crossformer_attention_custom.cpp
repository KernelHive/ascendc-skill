
#include "crossformer_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {
static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CrossformerAttentionCustomTilingData tiling;

    // Expected:
    // attn: [B,H,N,N]
    // v:    [B,H,N,Dh]
    // w:    [C,C]  where C=H*Dh (PyTorch Linear weight)
    // b:    [C]
    auto attn_shape = context->GetInputShape(0)->GetStorageShape();
    auto v_shape    = context->GetInputShape(1)->GetStorageShape();
    auto w_shape    = context->GetInputShape(2)->GetStorageShape();
    auto b_shape    = context->GetInputShape(3)->GetStorageShape();

    if (attn_shape.GetDimNum() != 4 || v_shape.GetDimNum() != 4 ||
        w_shape.GetDimNum() != 2 || b_shape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t B  = static_cast<uint32_t>(attn_shape.GetDim(0));
    const uint32_t H  = static_cast<uint32_t>(attn_shape.GetDim(1));
    const uint32_t N  = static_cast<uint32_t>(attn_shape.GetDim(2));
    const uint32_t N2 = static_cast<uint32_t>(attn_shape.GetDim(3));
    if (B == 0 || H == 0 || N == 0 || N2 == 0) return ge::GRAPH_FAILED;
    if (N != N2) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(v_shape.GetDim(0)) != B ||
        static_cast<uint32_t>(v_shape.GetDim(1)) != H ||
        static_cast<uint32_t>(v_shape.GetDim(2)) != N) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t Dh = static_cast<uint32_t>(v_shape.GetDim(3));
    if (Dh == 0) return ge::GRAPH_FAILED;

    const uint32_t C = H * Dh;

    // PyTorch Linear: weight is [out_features, in_features] = [C, C]
    if (static_cast<uint32_t>(w_shape.GetDim(0)) != C ||
        static_cast<uint32_t>(w_shape.GetDim(1)) != C) {
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint32_t>(b_shape.GetDim(0)) != C) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t totalRows = B * N;

    // Choose core number similar to successful example: moderate parallelism without instability.
    uint32_t idealCores = 24U;
    uint32_t coreNum = std::min<uint32_t>(idealCores, std::max<uint32_t>(1U, totalRows / 2U));
    coreNum = std::min<uint32_t>(coreNum, totalRows);
    if (coreNum == 0) coreNum = 1;
    context->SetBlockDim(coreNum);

    uint32_t blockRows = CeilDivU32(totalRows, coreNum);

    // UB Dh tile: keep conservative (float buffers).
    // Kernel uses 3*dhTile floats per head (attnRow, vRow, ctxRow).
    uint32_t dhTile = 64U;
    if (dhTile > Dh) dhTile = Dh;
    if (dhTile == 0) dhTile = 1;

    tiling.set_B(B);
    tiling.set_H(H);
    tiling.set_N(N);
    tiling.set_Dh(Dh);
    tiling.set_C(C);
    tiling.set_totalRows(totalRows);
    tiling.set_blockRows(blockRows);
    tiling.set_dhTile(dhTile);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class CrossformerAttentionCustom : public OpDef {
public:
    explicit CrossformerAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("attn")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("proj_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("proj_bias")
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
OP_ADD(CrossformerAttentionCustom);
} // namespace ops
