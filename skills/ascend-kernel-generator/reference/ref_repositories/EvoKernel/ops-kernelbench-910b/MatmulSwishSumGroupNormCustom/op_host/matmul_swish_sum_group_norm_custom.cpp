
#include "matmul_swish_sum_group_norm_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Specialized fixed contract:
// x: [32768,1024] float32 contiguous
// w: [4096,1024]  float32 contiguous (Linear weight [out,in] == [N,K])
// linear_bias: [4096] float32 contiguous (Linear bias)
// add_bias:    [4096] float32 contiguous (post-swish add bias)
// gamma/beta:  [4096] float32 contiguous (GroupNorm affine)
// num_groups: scalar int32 tensor ([] or [1]) expected 64
// eps: scalar float32 tensor ([] or [1])
// y: [32768,4096] float32 contiguous
static constexpr uint32_t M = 32768;
static constexpr uint32_t K = 1024;
static constexpr uint32_t N = 4096;
static constexpr uint32_t G_EXPECT = 64;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto sx   = context->GetInputShape(0);
    auto sw   = context->GetInputShape(1);
    auto slb  = context->GetInputShape(2);
    auto sab  = context->GetInputShape(3);
    auto sga  = context->GetInputShape(4);
    auto sbe  = context->GetInputShape(5);
    auto sng  = context->GetInputShape(6);
    auto seps = context->GetInputShape(7);
    auto sy   = context->GetOutputShape(0);

    if (sx == nullptr || sw == nullptr || slb == nullptr || sab == nullptr ||
        sga == nullptr || sbe == nullptr || sng == nullptr || seps == nullptr || sy == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& x     = sx->GetOriginShape();
    const auto& w     = sw->GetOriginShape();
    const auto& lb    = slb->GetOriginShape();
    const auto& ab    = sab->GetOriginShape();
    const auto& gamma = sga->GetOriginShape();
    const auto& beta  = sbe->GetOriginShape();
    const auto& ng    = sng->GetOriginShape();
    const auto& eps   = seps->GetOriginShape();

    if (x.GetDimNum() != 2 || w.GetDimNum() != 2) return ge::GRAPH_FAILED;
    if (lb.GetDimNum() != 1 || ab.GetDimNum() != 1 || gamma.GetDimNum() != 1 || beta.GetDimNum() != 1)
        return ge::GRAPH_FAILED;

    const bool ngOk  = (ng.GetDimNum() == 0)  || (ng.GetDimNum() == 1 && (uint32_t)ng.GetDim(0) == 1);
    const bool epsOk = (eps.GetDimNum() == 0) || (eps.GetDimNum() == 1 && (uint32_t)eps.GetDim(0) == 1);
    if (!ngOk || !epsOk) return ge::GRAPH_FAILED;

    if ((uint32_t)x.GetDim(0) != M || (uint32_t)x.GetDim(1) != K) return ge::GRAPH_FAILED;
    if ((uint32_t)w.GetDim(0) != N || (uint32_t)w.GetDim(1) != K) return ge::GRAPH_FAILED;
    if ((uint32_t)lb.GetDim(0) != N) return ge::GRAPH_FAILED;
    if ((uint32_t)ab.GetDim(0) != N) return ge::GRAPH_FAILED;
    if ((uint32_t)gamma.GetDim(0) != N) return ge::GRAPH_FAILED;
    if ((uint32_t)beta.GetDim(0) != N) return ge::GRAPH_FAILED;

    if (context->GetInputTensor(0)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(1)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(2)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(3)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(4)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(5)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(6)->GetDataType() != ge::DT_INT32) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(7)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;

    MatmulSwishSumGroupNormCustomTilingData tiling;
    tiling.set_totalX(sx->GetStorageShape().GetShapeSize());
    tiling.set_totalW(sw->GetStorageShape().GetShapeSize());
    tiling.set_totalLinearBias(slb->GetStorageShape().GetShapeSize());
    tiling.set_totalAddBias(sab->GetStorageShape().GetShapeSize());
    tiling.set_totalGamma(sga->GetStorageShape().GetShapeSize());
    tiling.set_totalBeta(sbe->GetStorageShape().GetShapeSize());
    tiling.set_totalNumGroups(sng->GetStorageShape().GetShapeSize());
    tiling.set_totalEps(seps->GetStorageShape().GetShapeSize());
    tiling.set_totalY(sy->GetStorageShape().GetShapeSize());

    tiling.set_M(M);
    tiling.set_K(K);
    tiling.set_N(N);
    tiling.set_totalElems(M * N);

    tiling.set_G(G_EXPECT);
    tiling.set_groupSize(N / G_EXPECT); // 64

    // Parallelize over rows.
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

class MatmulSwishSumGroupNormCustom : public OpDef {
public:
    explicit MatmulSwishSumGroupNormCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("linear_bias")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("add_bias")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("beta")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("num_groups")
            .ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("eps")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MatmulSwishSumGroupNormCustom);

} // namespace ops
