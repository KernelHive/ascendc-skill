
#include "shuffle_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace optiling {
static constexpr uint32_t BLOCK_SIZE = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ShuffleAttentionCustomTilingData tiling;

    // Contract:
    // x0     : [BG, C2g, H, W]  where BG=B*G and C2g=C/(2*G)
    // x1     : [BG, C2g, H, W]
    // gate_c : [BG, C2g, 1, 1]  (already sigmoid'ed)
    // s_norm : [BG, C2g, H, W]  (GroupNorm(x1))
    // sweight: [1,  C2g, 1, 1]
    // sbias  : [1,  C2g, 1, 1]
    // y      : [B, C, H, W]     (final, after correct channel_shuffle(groups=2))

    auto x0_shape = context->GetInputShape(0)->GetStorageShape();
    auto x1_shape = context->GetInputShape(1)->GetStorageShape();
    auto gc_shape = context->GetInputShape(2)->GetStorageShape();
    auto sn_shape = context->GetInputShape(3)->GetStorageShape();
    auto sw_shape = context->GetInputShape(4)->GetStorageShape();
    auto sb_shape = context->GetInputShape(5)->GetStorageShape();

    if (x0_shape.GetDimNum() != 4 || x1_shape.GetDimNum() != 4 ||
        gc_shape.GetDimNum() != 4 || sn_shape.GetDimNum() != 4 ||
        sw_shape.GetDimNum() != 4 || sb_shape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    int64_t BG = x0_shape.GetDim(0);
    int64_t C2g = x0_shape.GetDim(1);
    int64_t H = x0_shape.GetDim(2);
    int64_t W = x0_shape.GetDim(3);
    if (BG <= 0 || C2g <= 0 || H <= 0 || W <= 0) return ge::GRAPH_FAILED;

    if (x1_shape.GetDim(0) != BG || x1_shape.GetDim(1) != C2g || x1_shape.GetDim(2) != H || x1_shape.GetDim(3) != W)
        return ge::GRAPH_FAILED;
    if (sn_shape.GetDim(0) != BG || sn_shape.GetDim(1) != C2g || sn_shape.GetDim(2) != H || sn_shape.GetDim(3) != W)
        return ge::GRAPH_FAILED;

    if (gc_shape.GetDim(0) != BG || gc_shape.GetDim(1) != C2g || gc_shape.GetDim(2) != 1 || gc_shape.GetDim(3) != 1)
        return ge::GRAPH_FAILED;

    auto check_param = [&](const gert::Shape& s)->bool {
        return (s.GetDim(0) == 1 && s.GetDim(1) == C2g && s.GetDim(2) == 1 && s.GetDim(3) == 1);
    };
    if (!check_param(sw_shape) || !check_param(sb_shape)) return ge::GRAPH_FAILED;

    // Infer B and G: must have BG divisible by G, and C must be 2*G*C2g.
    // We don't get G explicitly from inputs; but module uses fixed G and framework will build accordingly.
    // For safety, we store only BG and C2g in inputs; however output requires B and C.
    // We can infer G by requiring C2g==channel/(2*G) is unknown. In this custom op, we assume G is provided
    // by framework as an attribute? Not available here. Therefore we infer G=8 by default? Not acceptable.
    // Solution: treat B and G as derived from output shape requested by framework.
    // In custom op ABI, output shape is inferred by python binding allocation, so context can read output shape.
    auto y_shape = context->GetOutputShape(0)->GetStorageShape();
    if (y_shape.GetDimNum() != 4) return ge::GRAPH_FAILED;

    int64_t B = y_shape.GetDim(0);
    int64_t C = y_shape.GetDim(1);
    int64_t yH = y_shape.GetDim(2);
    int64_t yW = y_shape.GetDim(3);
    if (B <= 0 || C <= 0 || yH != H || yW != W) return ge::GRAPH_FAILED;

    if (BG % B != 0) return ge::GRAPH_FAILED;
    int64_t G = BG / B;
    if (G <= 0) return ge::GRAPH_FAILED;

    if (C != 2 * G * C2g) return ge::GRAPH_FAILED;

    uint64_t HW64 = static_cast<uint64_t>(H) * static_cast<uint64_t>(W);
    if (HW64 == 0 || HW64 > 0xFFFFFFFFu) return ge::GRAPH_FAILED;
    uint32_t HW = static_cast<uint32_t>(HW64);

    uint64_t yElems64 = static_cast<uint64_t>(B) * static_cast<uint64_t>(C) * static_cast<uint64_t>(HW);
    if (yElems64 == 0 || yElems64 > 0xFFFFFFFFu) return ge::GRAPH_FAILED;
    uint32_t yElems = static_cast<uint32_t>(yElems64);

    // Multi-core. Kernel computes runtime split from yElems.
    uint32_t coreNum = 8;
    context->SetBlockDim(coreNum);

    const uint32_t sizeofdatatype = 4;
    const uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype; // 8 floats per 32B

    // UB conservative: 2 temp vectors (affine, gate_s) + out tile
    uint32_t ub_block_num = 1024; // 32KB
    if (ub_block_num % 2 != 0) ub_block_num -= 1;
    uint32_t tileElems = ub_block_num * ALIGN_NUM;
    if (tileElems < ALIGN_NUM * 8) tileElems = ALIGN_NUM * 8;

    uint32_t tileNum = (yElems + tileElems - 1) / tileElems;
    if (tileNum == 0) tileNum = 1;
    uint32_t lastTileElems = yElems - (tileNum - 1) * tileElems;
    if (lastTileElems == 0) lastTileElems = tileElems;

    tiling.set_B(static_cast<uint32_t>(B));
    tiling.set_C(static_cast<uint32_t>(C));
    tiling.set_H(static_cast<uint32_t>(H));
    tiling.set_W(static_cast<uint32_t>(W));
    tiling.set_G(static_cast<uint32_t>(G));
    tiling.set_C2g(static_cast<uint32_t>(C2g));
    tiling.set_HW(HW);
    tiling.set_yElems(yElems);

    tiling.set_coreNum(coreNum);
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
class ShuffleAttentionCustom : public OpDef {
public:
    explicit ShuffleAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("x0").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x1").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gate_c").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("s_norm").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sweight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sbias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(ShuffleAttentionCustom);
} // namespace ops
