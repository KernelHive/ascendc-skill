
#include "lstm_cn_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <algorithm>

namespace optiling {

static constexpr uint32_t B_EXP = 10;
static constexpr uint32_t S_EXP = 512;
static constexpr uint32_t I_EXP = 128;
static constexpr uint32_t H_EXP = 256;
static constexpr uint32_t L_EXP = 6;

static inline uint64_t GateRows() { return (uint64_t)L_EXP * 4ULL * (uint64_t)H_EXP; } // 6144

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    auto sx  = context->GetInputShape(0);
    auto sh0 = context->GetInputShape(1);
    auto sc0 = context->GetInputShape(2);
    auto swi = context->GetInputShape(3);
    auto swh = context->GetInputShape(4);
    auto sbi = context->GetInputShape(5);
    auto sbh = context->GetInputShape(6);
    if (sx == nullptr || sh0 == nullptr || sc0 == nullptr || swi == nullptr || swh == nullptr || sbi == nullptr || sbh == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& x   = sx->GetOriginShape();
    const auto& h0  = sh0->GetOriginShape();
    const auto& c0  = sc0->GetOriginShape();
    const auto& wih = swi->GetOriginShape();
    const auto& whh = swh->GetOriginShape();
    const auto& bih = sbi->GetOriginShape();
    const auto& bhh = sbh->GetOriginShape();

    if (x.GetDimNum() != 3 || h0.GetDimNum() != 3 || c0.GetDimNum() != 3) return ge::GRAPH_FAILED;
    if (wih.GetDimNum() != 2 || whh.GetDimNum() != 2) return ge::GRAPH_FAILED;
    if (bih.GetDimNum() != 1 || bhh.GetDimNum() != 1) return ge::GRAPH_FAILED;

    if ((uint32_t)x.GetDim(0) != B_EXP || (uint32_t)x.GetDim(1) != S_EXP || (uint32_t)x.GetDim(2) != I_EXP) return ge::GRAPH_FAILED;
    if ((uint32_t)h0.GetDim(0) != L_EXP || (uint32_t)h0.GetDim(1) != B_EXP || (uint32_t)h0.GetDim(2) != H_EXP) return ge::GRAPH_FAILED;
    if ((uint32_t)c0.GetDim(0) != L_EXP || (uint32_t)c0.GetDim(1) != B_EXP || (uint32_t)c0.GetDim(2) != H_EXP) return ge::GRAPH_FAILED;

    const uint64_t rows = GateRows();
    if ((uint64_t)wih.GetDim(0) != rows || (uint32_t)wih.GetDim(1) != H_EXP) return ge::GRAPH_FAILED;
    if ((uint64_t)whh.GetDim(0) != rows || (uint32_t)whh.GetDim(1) != H_EXP) return ge::GRAPH_FAILED;
    if ((uint64_t)bih.GetDim(0) != rows) return ge::GRAPH_FAILED;
    if ((uint64_t)bhh.GetDim(0) != rows) return ge::GRAPH_FAILED;

    auto so = context->GetOutputShape(0);
    if (so == nullptr) return ge::GRAPH_FAILED;
    const auto& out = so->GetOriginShape();
    if (out.GetDimNum() != 3 ||
        (uint32_t)out.GetDim(0) != L_EXP ||
        (uint32_t)out.GetDim(1) != B_EXP ||
        (uint32_t)out.GetDim(2) != H_EXP) return ge::GRAPH_FAILED;

    for (int i = 0; i < 7; ++i) {
        if (context->GetInputTensor(i)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    }

    LstmCnCustomTilingData tiling;
    tiling.set_totalX(sx->GetStorageShape().GetShapeSize());
    tiling.set_totalH0(sh0->GetStorageShape().GetShapeSize());
    tiling.set_totalC0(sc0->GetStorageShape().GetShapeSize());
    tiling.set_totalWih(swi->GetStorageShape().GetShapeSize());
    tiling.set_totalWhh(swh->GetStorageShape().GetShapeSize());
    tiling.set_totalBih(sbi->GetStorageShape().GetShapeSize());
    tiling.set_totalBhh(sbh->GetStorageShape().GetShapeSize());
    tiling.set_totalCn(so->GetStorageShape().GetShapeSize());

    tiling.set_B(B_EXP);
    tiling.set_S(S_EXP);
    tiling.set_I(I_EXP);
    tiling.set_H(H_EXP);
    tiling.set_L(L_EXP);

    const uint32_t totalB = B_EXP;
    uint32_t blockDim = 5; // slightly higher occupancy than 4, still conservative for B=10
    if (blockDim > totalB) blockDim = totalB;
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    uint32_t blockB = (totalB + blockDim - 1) / blockDim;
    if (blockB == 0) blockB = 1;
    tiling.set_totalB(totalB);
    tiling.set_blockB(blockB);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class LstmCnCustom : public OpDef {
public:
    explicit LstmCnCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("h0").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("c0").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("w_ih").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w_hh").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_ih").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_hh").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("c_n").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LstmCnCustom);

} // namespace ops
