
#include "matmul_group_norm_leaky_relu_sum_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Fixed specialized contract for performance/simplicity.
// x:    [1024,8192] float32 contiguous
// w:    [8192,8192] float32 contiguous (Linear weight [out,in] == [N,K])
// bias: [8192] float32 contiguous
// gamma:[8192] float32 contiguous
// beta: [8192] float32 contiguous
// eps: scalar float32 tensor ([] or [1])
// negative_slope: scalar float32 tensor ([] or [1])
// y:    [1024,8192] float32
// GroupNorm: num_groups=512 => groupSize=16
static constexpr uint32_t M_FIX = 1024;
static constexpr uint32_t K_FIX = 8192;
static constexpr uint32_t N_FIX = 8192;
static constexpr uint32_t G_FIX = 512;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto sx = context->GetInputShape(0);
    auto sw = context->GetInputShape(1);
    auto sbias = context->GetInputShape(2);
    auto sgamma = context->GetInputShape(3);
    auto sbeta = context->GetInputShape(4);
    auto seps = context->GetInputShape(5);
    auto sns = context->GetInputShape(6);

    if (sx == nullptr || sw == nullptr || sbias == nullptr || sgamma == nullptr ||
        sbeta == nullptr || seps == nullptr || sns == nullptr ||
        context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& x = sx->GetOriginShape();
    const auto& w = sw->GetOriginShape();
    const auto& bias = sbias->GetOriginShape();
    const auto& gamma = sgamma->GetOriginShape();
    const auto& beta = sbeta->GetOriginShape();
    const auto& eps = seps->GetOriginShape();
    const auto& ns = sns->GetOriginShape();

    if (x.GetDimNum() != 2 || w.GetDimNum() != 2) return ge::GRAPH_FAILED;
    if (bias.GetDimNum() != 1 || gamma.GetDimNum() != 1 || beta.GetDimNum() != 1) return ge::GRAPH_FAILED;

    // scalar tensors can be [] or [1]
    const bool epsOk = (eps.GetDimNum() == 0) || (eps.GetDimNum() == 1 && (uint32_t)eps.GetDim(0) == 1);
    const bool nsOk  = (ns.GetDimNum() == 0)  || (ns.GetDimNum() == 1 && (uint32_t)ns.GetDim(0) == 1);
    if (!epsOk || !nsOk) return ge::GRAPH_FAILED;

    if ((uint32_t)x.GetDim(0) != M_FIX || (uint32_t)x.GetDim(1) != K_FIX) return ge::GRAPH_FAILED;
    if ((uint32_t)w.GetDim(0) != N_FIX || (uint32_t)w.GetDim(1) != K_FIX) return ge::GRAPH_FAILED;
    if ((uint32_t)bias.GetDim(0) != N_FIX) return ge::GRAPH_FAILED;
    if ((uint32_t)gamma.GetDim(0) != N_FIX) return ge::GRAPH_FAILED;
    if ((uint32_t)beta.GetDim(0) != N_FIX) return ge::GRAPH_FAILED;

    if (context->GetInputTensor(0)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(1)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(2)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(3)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(4)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(5)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(6)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;

    if ((N_FIX % G_FIX) != 0u) return ge::GRAPH_FAILED;

    MatmulGroupNormLeakyReluSumCustomTilingData tiling;
    tiling.set_totalX(sx->GetStorageShape().GetShapeSize());
    tiling.set_totalW(sw->GetStorageShape().GetShapeSize());
    tiling.set_totalBias(sbias->GetStorageShape().GetShapeSize());
    tiling.set_totalGamma(sgamma->GetStorageShape().GetShapeSize());
    tiling.set_totalBeta(sbeta->GetStorageShape().GetShapeSize());
    tiling.set_totalEps(seps->GetStorageShape().GetShapeSize());
    tiling.set_totalNegativeSlope(sns->GetStorageShape().GetShapeSize());
    tiling.set_totalY(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());

    tiling.set_M(M_FIX);
    tiling.set_K(K_FIX);
    tiling.set_N(N_FIX);
    tiling.set_G(G_FIX);
    tiling.set_groupSize(N_FIX / G_FIX); // 16

    // Parallelize over rows (m). Each core handles a contiguous m-range.
    context->SetBlockDim(64);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class MatmulGroupNormLeakyReluSumCustom : public OpDef {
public:
    explicit MatmulGroupNormLeakyReluSumCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("beta").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("eps").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("negative_slope").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MatmulGroupNormLeakyReluSumCustom);

} // namespace ops
