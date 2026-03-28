
#include "mobile_vi_tv2_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MobileViTv2AttentionCustomTilingData tiling;

    const auto& ishape = context->GetInputShape(0)->GetStorageShape();
    const auto& kshape = context->GetInputShape(1)->GetStorageShape();
    const auto& vshape = context->GetInputShape(2)->GetStorageShape();
    const auto& wshape = context->GetInputShape(3)->GetStorageShape();
    const auto& bshape = context->GetInputShape(4)->GetStorageShape();

    if (ishape.GetDimNum() != 3 || kshape.GetDimNum() != 3 || vshape.GetDimNum() != 3 ||
        wshape.GetDimNum() != 2 || bshape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    uint32_t bs = static_cast<uint32_t>(ishape.GetDim(0));
    uint32_t nq = static_cast<uint32_t>(ishape.GetDim(1));
    uint32_t one = static_cast<uint32_t>(ishape.GetDim(2));
    if (bs == 0 || nq == 0 || one != 1) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(kshape.GetDim(0)) != bs || static_cast<uint32_t>(kshape.GetDim(1)) != nq) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(vshape.GetDim(0)) != bs || static_cast<uint32_t>(vshape.GetDim(1)) != nq) return ge::GRAPH_FAILED;

    uint32_t d = static_cast<uint32_t>(kshape.GetDim(2));
    if (d == 0) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(vshape.GetDim(2)) != d) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(wshape.GetDim(0)) != d || static_cast<uint32_t>(wshape.GetDim(1)) != d) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(bshape.GetDim(0)) != d) return ge::GRAPH_FAILED;

    tiling.set_bs(bs);
    tiling.set_nq(nq);
    tiling.set_d(d);

    // Single-tile softmax for typical nq=49 to remove loop/control overhead
    uint32_t qTile = nq;
    if (qTile == 0) qTile = 1;
    tiling.set_qTile(qTile);

    uint32_t dTile = 128;
    if (dTile > d) dTile = d;
    if (dTile == 0) dTile = 1;
    if (dTile % 8 != 0) dTile = ((dTile + 7) / 8) * 8;
    if (dTile > d) dTile = d;
    tiling.set_dTile(dTile);

    uint32_t doTile = 16; // compact UB panel; good occupancy via doGroups mapping
    if (doTile > d) doTile = d;
    if (doTile == 0) doTile = 1;
    if (doTile % 8 != 0) doTile = ((doTile + 7) / 8) * 8;
    if (doTile > d) doTile = d;
    tiling.set_doTile(doTile);

    // Parallelize across (bs * doGroups) for occupancy
    uint32_t doGroups = (d + doTile - 1) / doTile;
    uint32_t work = bs * doGroups;
    uint32_t block_dim = std::min<uint32_t>(work, 48);
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
class MobileViTv2AttentionCustom : public OpDef {
public:
    explicit MobileViTv2AttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("i").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w_o").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_o").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MobileViTv2AttentionCustom);
} // namespace ops
