
#include "gemm_group_norm_swish_multiply_swish_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Fixed specialized contract for performance & simplicity:
// x: [1024,8192] float32 contiguous
// w: [8192,8192] float32 contiguous (Linear weight [out,in] == [N,K])
// b: [8192] float32 contiguous
// gamma: [8192] float32 contiguous
// beta:  [8192] float32 contiguous
// mul_w: [8192] float32 contiguous
// num_groups: scalar int32 tensor ([] or [1]) expected 256
// eps: scalar float32 tensor ([] or [1])
// y: [1024,8192] float32 contiguous
static constexpr uint32_t M = 1024;
static constexpr uint32_t K = 8192;
static constexpr uint32_t N = 8192;
static constexpr uint32_t G_EXPECT = 256;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto sx   = context->GetInputShape(0);
    auto sw   = context->GetInputShape(1);
    auto sb   = context->GetInputShape(2);
    auto sga  = context->GetInputShape(3);
    auto sbe  = context->GetInputShape(4);
    auto smw  = context->GetInputShape(5);
    auto sng  = context->GetInputShape(6);
    auto seps = context->GetInputShape(7);

    if (sx == nullptr || sw == nullptr || sb == nullptr || sga == nullptr || sbe == nullptr ||
        smw == nullptr || sng == nullptr || seps == nullptr || context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& x     = sx->GetOriginShape();
    const auto& w     = sw->GetOriginShape();
    const auto& b     = sb->GetOriginShape();
    const auto& gamma = sga->GetOriginShape();
    const auto& beta  = sbe->GetOriginShape();
    const auto& mulw  = smw->GetOriginShape();
    const auto& ng    = sng->GetOriginShape();
    const auto& eps   = seps->GetOriginShape();

    if (x.GetDimNum() != 2 || w.GetDimNum() != 2 || b.GetDimNum() != 1 ||
        gamma.GetDimNum() != 1 || beta.GetDimNum() != 1 || mulw.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const bool ngOk  = (ng.GetDimNum() == 0)  || (ng.GetDimNum() == 1 && (uint32_t)ng.GetDim(0) == 1);
    const bool epsOk = (eps.GetDimNum() == 0) || (eps.GetDimNum() == 1 && (uint32_t)eps.GetDim(0) == 1);
    if (!ngOk || !epsOk) return ge::GRAPH_FAILED;

    if ((uint32_t)x.GetDim(0) != M || (uint32_t)x.GetDim(1) != K) return ge::GRAPH_FAILED;
    if ((uint32_t)w.GetDim(0) != N || (uint32_t)w.GetDim(1) != K) return ge::GRAPH_FAILED;
    if ((uint32_t)b.GetDim(0) != N) return ge::GRAPH_FAILED;
    if ((uint32_t)gamma.GetDim(0) != N) return ge::GRAPH_FAILED;
    if ((uint32_t)beta.GetDim(0) != N) return ge::GRAPH_FAILED;
    if ((uint32_t)mulw.GetDim(0) != N) return ge::GRAPH_FAILED;

    if (context->GetInputTensor(0)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(1)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(2)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(3)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(4)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(5)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(6)->GetDataType() != ge::DT_INT32) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(7)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;

    GemmGroupNormSwishMultiplySwishCustomTilingData tiling;
    tiling.set_totalX(sx->GetStorageShape().GetShapeSize());
    tiling.set_totalW(sw->GetStorageShape().GetShapeSize());
    tiling.set_totalB(sb->GetStorageShape().GetShapeSize());
    tiling.set_totalGamma(sga->GetStorageShape().GetShapeSize());
    tiling.set_totalBeta(sbe->GetStorageShape().GetShapeSize());
    tiling.set_totalMulW(smw->GetStorageShape().GetShapeSize());
    tiling.set_totalNumGroups(sng->GetStorageShape().GetShapeSize());
    tiling.set_totalEps(seps->GetStorageShape().GetShapeSize());
    tiling.set_totalY(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());

    tiling.set_M(M);
    tiling.set_K(K);
    tiling.set_N(N);
    tiling.set_totalElems(M * N);

    tiling.set_G(G_EXPECT);
    tiling.set_groupSize(N / G_EXPECT); // 32

    // Parallelize over rows (M). Kernel computes per-row per-group stats once and normalizes.
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

class GemmGroupNormSwishMultiplySwishCustom : public OpDef {
public:
    explicit GemmGroupNormSwishMultiplySwishCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("beta")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mul_w")
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

OP_ADD(GemmGroupNormSwishMultiplySwishCustom);

} // namespace ops
