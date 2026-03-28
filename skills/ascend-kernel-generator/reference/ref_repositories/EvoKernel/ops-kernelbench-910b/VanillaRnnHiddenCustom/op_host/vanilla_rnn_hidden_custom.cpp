
#include "vanilla_rnn_hidden_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <algorithm>

namespace optiling {

// Specialized benchmark contract (must match Python binding + model packing)
static constexpr uint32_t T_EXP = 256;
static constexpr uint32_t B_EXP = 8;
static constexpr uint32_t I_EXP = 1024;
static constexpr uint32_t H_EXP = 256;
static constexpr uint32_t O_EXP = 128;
static constexpr uint32_t K_EXP = I_EXP + H_EXP; // 1280

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    auto sx  = context->GetInputShape(0);
    auto sh0 = context->GetInputShape(1);
    auto swi = context->GetInputShape(2);
    auto sbi = context->GetInputShape(3);
    auto swo = context->GetInputShape(4);
    auto sbo = context->GetInputShape(5);
    if (sx == nullptr || sh0 == nullptr || swi == nullptr || sbi == nullptr || swo == nullptr || sbo == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& x   = sx->GetOriginShape();   // [T,B,I]
    const auto& h0  = sh0->GetOriginShape();  // [B,H]
    const auto& wi2h= swi->GetOriginShape();  // [H,K]
    const auto& bi2h= sbi->GetOriginShape();  // [H]
    const auto& wh2o= swo->GetOriginShape();  // [O,H]
    const auto& bh2o= sbo->GetOriginShape();  // [O]

    if (x.GetDimNum() != 3 || h0.GetDimNum() != 2) return ge::GRAPH_FAILED;
    if (wi2h.GetDimNum() != 2 || bi2h.GetDimNum() != 1) return ge::GRAPH_FAILED;
    if (wh2o.GetDimNum() != 2 || bh2o.GetDimNum() != 1) return ge::GRAPH_FAILED;

    if ((uint32_t)x.GetDim(0) != T_EXP || (uint32_t)x.GetDim(1) != B_EXP || (uint32_t)x.GetDim(2) != I_EXP) return ge::GRAPH_FAILED;
    if ((uint32_t)h0.GetDim(0) != B_EXP || (uint32_t)h0.GetDim(1) != H_EXP) return ge::GRAPH_FAILED;

    if ((uint32_t)wi2h.GetDim(0) != H_EXP || (uint32_t)wi2h.GetDim(1) != K_EXP) return ge::GRAPH_FAILED;
    if ((uint32_t)bi2h.GetDim(0) != H_EXP) return ge::GRAPH_FAILED;

    if ((uint32_t)wh2o.GetDim(0) != O_EXP || (uint32_t)wh2o.GetDim(1) != H_EXP) return ge::GRAPH_FAILED;
    if ((uint32_t)bh2o.GetDim(0) != O_EXP) return ge::GRAPH_FAILED;

    auto so = context->GetOutputShape(0);
    if (so == nullptr) return ge::GRAPH_FAILED;
    const auto& y = so->GetOriginShape();
    if (y.GetDimNum() != 3 ||
        (uint32_t)y.GetDim(0) != T_EXP ||
        (uint32_t)y.GetDim(1) != B_EXP ||
        (uint32_t)y.GetDim(2) != O_EXP) return ge::GRAPH_FAILED;

    for (int i = 0; i < 6; ++i) {
        if (context->GetInputTensor(i)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    }

    VanillaRnnHiddenCustomTilingData tiling;
    tiling.set_totalX(sx->GetStorageShape().GetShapeSize());
    tiling.set_totalH0(sh0->GetStorageShape().GetShapeSize());
    tiling.set_totalWi2h(swi->GetStorageShape().GetShapeSize());
    tiling.set_totalBi2h(sbi->GetStorageShape().GetShapeSize());
    tiling.set_totalWh2o(swo->GetStorageShape().GetShapeSize());
    tiling.set_totalBh2o(sbo->GetStorageShape().GetShapeSize());
    tiling.set_totalY(so->GetStorageShape().GetShapeSize());

    tiling.set_T(T_EXP);
    tiling.set_B(B_EXP);
    tiling.set_I(I_EXP);
    tiling.set_H(H_EXP);
    tiling.set_O(O_EXP);
    tiling.set_K(K_EXP);

    // Parallelism: split by batch; B=8 so use up to 8 blocks (more occupancy than baseline 4).
    // Each block processes blockB batch items sequentially (hidden state dependency is within b, not across b).
    const uint32_t totalB = B_EXP;
    uint32_t blockDim = 8;
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

class VanillaRnnHiddenCustom : public OpDef {
public:
    explicit VanillaRnnHiddenCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("h0").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w_i2h").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_i2h").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w_h2o").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_h2o").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(VanillaRnnHiddenCustom);

} // namespace ops
