
#include "vi_t_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static inline uint32_t AlignUp(uint32_t x, uint32_t a) { return (x + a - 1) / a * a; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ViTAttentionCustomTilingData tiling;

    const auto& qshape = context->GetInputShape(0)->GetStorageShape();
    const auto& kshape = context->GetInputShape(1)->GetStorageShape();
    const auto& vshape = context->GetInputShape(2)->GetStorageShape();
    const auto& sshape = context->GetInputShape(3)->GetStorageShape();
    const auto& wshape = context->GetInputShape(4)->GetStorageShape();
    const auto& bshape = context->GetInputShape(5)->GetStorageShape();

    if (qshape.GetDimNum() != 4 || kshape.GetDimNum() != 4 || vshape.GetDimNum() != 4 ||
        wshape.GetDimNum() != 2 || bshape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    if (!(sshape.GetDimNum() == 0 || (sshape.GetDimNum() == 1 && static_cast<uint32_t>(sshape.GetDim(0)) == 1))) {
        return ge::GRAPH_FAILED;
    }

    uint32_t bs    = static_cast<uint32_t>(qshape.GetDim(0));
    uint32_t heads = static_cast<uint32_t>(qshape.GetDim(1));
    uint32_t nq    = static_cast<uint32_t>(qshape.GetDim(2));
    uint32_t d     = static_cast<uint32_t>(qshape.GetDim(3));
    if (bs == 0 || heads == 0 || nq == 0 || d == 0) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(kshape.GetDim(0)) != bs || static_cast<uint32_t>(kshape.GetDim(1)) != heads ||
        static_cast<uint32_t>(kshape.GetDim(2)) != nq || static_cast<uint32_t>(kshape.GetDim(3)) != d) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(vshape.GetDim(0)) != bs || static_cast<uint32_t>(vshape.GetDim(1)) != heads ||
        static_cast<uint32_t>(vshape.GetDim(2)) != nq || static_cast<uint32_t>(vshape.GetDim(3)) != d) return ge::GRAPH_FAILED;

    uint32_t c = heads * d;
    if (static_cast<uint32_t>(wshape.GetDim(0)) != c || static_cast<uint32_t>(wshape.GetDim(1)) != c) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(bshape.GetDim(0)) != c) return ge::GRAPH_FAILED;

    tiling.set_bs(bs);
    tiling.set_heads(heads);
    tiling.set_nq(nq);
    tiling.set_d(d);
    tiling.set_c(c);

    // projection tiling: larger ocTile reduces scalar loop overhead
    uint32_t icTile = 128;
    if (icTile > c) icTile = c;
    icTile = std::max<uint32_t>(8, AlignUp(icTile, 8));
    if (icTile > c) icTile = c;
    tiling.set_icTile(icTile);

    uint32_t ocTile = 64;
    if (ocTile > c) ocTile = c;
    ocTile = std::max<uint32_t>(1, AlignUp(ocTile, 8));
    if (ocTile > c) ocTile = c;
    tiling.set_ocTile(ocTile);

    // stage K/V for typical ViT N=49 to reduce scalar GM reads in DotQK and V accumulation
    tiling.set_kvStage((nq <= 64) ? 1u : 0u);

    // token-parallel mapping for fused projection correctness
    uint32_t total = bs * nq;
    uint32_t block_dim = std::min<uint32_t>(total, 48);
    if (block_dim == 0) block_dim = 1;
    context->SetBlockDim(block_dim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
class ViTAttentionCustom : public OpDef {
public:
    explicit ViTAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("proj_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("proj_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ViTAttentionCustom);
} // namespace ops
