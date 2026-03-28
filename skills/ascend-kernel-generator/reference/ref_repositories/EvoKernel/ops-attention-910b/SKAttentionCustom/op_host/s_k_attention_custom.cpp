
#include "sk_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <string.h>

namespace optiling {
static constexpr uint32_t BLOCK_SIZE = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SKAttentionCustomTilingData tiling;

    // attn:  [K, bs, C, 1, 1]
    // feats: [K, bs, C, H, W]
    auto attn_shape  = context->GetInputShape(0)->GetStorageShape();
    auto feats_shape = context->GetInputShape(1)->GetStorageShape();

    if (attn_shape.GetDimNum() != 5 || feats_shape.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t K  = static_cast<uint32_t>(attn_shape.GetDim(0));
    const uint32_t bs = static_cast<uint32_t>(attn_shape.GetDim(1));
    const uint32_t C  = static_cast<uint32_t>(attn_shape.GetDim(2));
    if (static_cast<uint32_t>(attn_shape.GetDim(3)) != 1 || static_cast<uint32_t>(attn_shape.GetDim(4)) != 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t fK  = static_cast<uint32_t>(feats_shape.GetDim(0));
    const uint32_t fbs = static_cast<uint32_t>(feats_shape.GetDim(1));
    const uint32_t fC  = static_cast<uint32_t>(feats_shape.GetDim(2));
    const uint32_t H   = static_cast<uint32_t>(feats_shape.GetDim(3));
    const uint32_t W   = static_cast<uint32_t>(feats_shape.GetDim(4));

    if (K == 0 || bs == 0 || C == 0 || H == 0 || W == 0) return ge::GRAPH_FAILED;
    if (fK != K || fbs != bs || fC != C) return ge::GRAPH_FAILED;

    const uint32_t hw = H * W;
    const uint32_t totalOut = bs * C * hw;

    const uint32_t sizeofdatatype = 4; // float32
    const uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype; // 8 floats per 32B

    // Choose a safe core count. Keep simple and stable.
    uint32_t coreNum = 1;
    context->SetBlockDim(coreNum);

    // Per-core split on flattened output [bs*C*H*W]
    uint32_t per = (totalOut + coreNum - 1) / coreNum;
    uint32_t coreStart = 0;
    uint32_t coreCount = totalOut; // since coreNum=1

    // Tile sizing: keep moderate UB footprint. We need 3 local tensors (feats, acc, out) + small scalar work.
    // Use 2048 * 8 = 16384 elems = 64KB per tensor => 3 tensors ~192KB, may be high; choose 1024 blocks => 32KB per tensor => ~96KB.
    uint32_t ub_block_num = 1024; // blocks of 32B
    if (ub_block_num % 2 != 0) ub_block_num -= 1;

    uint32_t tileElems = ub_block_num * ALIGN_NUM; // aligned
    if (tileElems == 0) tileElems = ALIGN_NUM;

    uint32_t tileNum = (coreCount + tileElems - 1) / tileElems;
    if (tileNum == 0) tileNum = 1;
    uint32_t lastTileElems = coreCount - (tileNum - 1) * tileElems;
    if (lastTileElems == 0) lastTileElems = tileElems;

    tiling.set_K(K);
    tiling.set_bs(bs);
    tiling.set_C(C);
    tiling.set_H(H);
    tiling.set_W(W);
    tiling.set_hw(hw);
    tiling.set_totalOut(totalOut);
    tiling.set_coreNum(coreNum);
    tiling.set_coreStart(coreStart);
    tiling.set_coreCount(coreCount);
    tiling.set_tileNum(tileNum);
    tiling.set_tileElems(tileElems);
    tiling.set_lastTileElems(lastTileElems);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class SKAttentionCustom : public OpDef {
public:
    explicit SKAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("attn")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("feats")
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
OP_ADD(SKAttentionCustom);
} // namespace ops
