
#include "block_sparse_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <stdint.h>

namespace optiling {

static inline uint32_t RoundDownPow2(uint32_t x) {
    if (x == 0) return 0;
    uint32_t p = 1;
    while ((p << 1) <= x) p <<= 1;
    return p;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    BlockSparseAttentionCustomTilingData tiling;

    auto q_shape = context->GetInputShape(0)->GetStorageShape();
    auto k_shape = context->GetInputShape(1)->GetStorageShape();
    auto v_shape = context->GetInputShape(2)->GetStorageShape();

    if (q_shape.GetDimNum() != 5 || k_shape.GetDimNum() != 5 || v_shape.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t B  = static_cast<uint32_t>(q_shape.GetDim(0));
    const uint32_t H  = static_cast<uint32_t>(q_shape.GetDim(1));
    const uint32_t NB = static_cast<uint32_t>(q_shape.GetDim(2));
    const uint32_t BS = static_cast<uint32_t>(q_shape.GetDim(3));
    const uint32_t DK = static_cast<uint32_t>(q_shape.GetDim(4));

    if (B == 0 || H == 0 || NB == 0 || BS == 0 || DK == 0) return ge::GRAPH_FAILED;

    for (int i = 0; i < 5; ++i) {
        if (static_cast<uint32_t>(k_shape.GetDim(i)) != static_cast<uint32_t>(q_shape.GetDim(i))) return ge::GRAPH_FAILED;
        if (static_cast<uint32_t>(v_shape.GetDim(i)) != static_cast<uint32_t>(q_shape.GetDim(i))) return ge::GRAPH_FAILED;
    }

    const uint32_t totalBlocks = B * H * NB;

    uint32_t coreNum = std::min<uint32_t>(std::max<uint32_t>(1u, totalBlocks), 24u);
    context->SetBlockDim(coreNum);

    // Prefer dkTile 128 then 64, else power-of-two <= 64.
    uint32_t dkTile = 128;
    if (DK < 128) dkTile = (DK >= 64) ? 64 : RoundDownPow2(DK);
    if (dkTile == 0) dkTile = DK;
    if (dkTile > 128) dkTile = 128;

    // Favor jTile=16 for BS=32 to reduce loop overhead with good vector length.
    uint32_t jTile = 16;
    if (BS != 32) {
        jTile = (BS >= 16) ? 16 : RoundDownPow2(BS);
        if (jTile == 0) jTile = BS;
    }

    uint32_t useFastPath32 = (BS == 32 && (DK % 32u) == 0u) ? 1u : 0u;

    tiling.set_B(B);
    tiling.set_H(H);
    tiling.set_NB(NB);
    tiling.set_BS(BS);
    tiling.set_DK(DK);
    tiling.set_totalBlocks(totalBlocks);
    tiling.set_coreNum(coreNum);
    tiling.set_dkTile(dkTile);
    tiling.set_jTile(jTile);
    tiling.set_useFastPath32(useFastPath32);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
class BlockSparseAttentionCustom : public OpDef {
public:
    explicit BlockSparseAttentionCustom(const char* name) : OpDef(name)
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
        this->Input("scale")
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
OP_ADD(BlockSparseAttentionCustom);
} // namespace ops
