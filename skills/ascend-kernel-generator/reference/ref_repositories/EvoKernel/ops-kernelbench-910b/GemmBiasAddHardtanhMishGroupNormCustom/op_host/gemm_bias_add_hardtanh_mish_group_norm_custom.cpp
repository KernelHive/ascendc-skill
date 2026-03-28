
#include "gemm_bias_add_hardtanh_mish_group_norm_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static constexpr uint32_t M_EXPECT = 1024;
static constexpr uint32_t K_EXPECT = 8192;
static constexpr uint32_t N_EXPECT = 8192;
static constexpr uint32_t G_EXPECT = 256;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GemmBiasAddHardtanhMishGroupNormCustomTilingData t;

    const auto xShape  = context->GetInputShape(0)->GetStorageShape();
    const auto wShape  = context->GetInputShape(1)->GetStorageShape();
    const auto lbShape = context->GetInputShape(2)->GetStorageShape();
    const auto bShape  = context->GetInputShape(3)->GetStorageShape();
    const auto ggShape = context->GetInputShape(4)->GetStorageShape();
    const auto gbShape = context->GetInputShape(5)->GetStorageShape();

    if (xShape.GetDimNum() != 2 || wShape.GetDimNum() != 2) return ge::GRAPH_FAILED;
    if (lbShape.GetDimNum() != 1 || bShape.GetDimNum() != 1 ||
        ggShape.GetDimNum() != 1 || gbShape.GetDimNum() != 1) return ge::GRAPH_FAILED;

    const uint32_t M  = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t K  = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t N  = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t wK = static_cast<uint32_t>(wShape.GetDim(1));
    if (wK != K) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(lbShape.GetDim(0)) != N) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(bShape.GetDim(0))  != N) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(ggShape.GetDim(0)) != N) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(gbShape.GetDim(0)) != N) return ge::GRAPH_FAILED;

    if (!(M == M_EXPECT && K == K_EXPECT && N == N_EXPECT)) return ge::GRAPH_FAILED;

    const uint32_t numGroups = G_EXPECT;
    const uint32_t groupSize = N_EXPECT / numGroups;
    if (groupSize * numGroups != N_EXPECT) return ge::GRAPH_FAILED;

    const uint64_t totalTasks64 = static_cast<uint64_t>(M) * static_cast<uint64_t>(numGroups);
    if (totalTasks64 == 0 || totalTasks64 > 0xFFFFFFFFull) return ge::GRAPH_FAILED;

    t.set_M(M);
    t.set_K(K);
    t.set_N(N);
    t.set_num_groups(numGroups);
    t.set_group_size(groupSize);
    t.set_total_tasks(static_cast<uint32_t>(totalTasks64));

    t.set_total_x(static_cast<uint32_t>(xShape.GetShapeSize()));
    t.set_total_w(static_cast<uint32_t>(wShape.GetShapeSize()));
    t.set_total_lin_bias(static_cast<uint32_t>(lbShape.GetShapeSize()));
    t.set_total_bias(static_cast<uint32_t>(bShape.GetShapeSize()));
    t.set_total_gamma(static_cast<uint32_t>(ggShape.GetShapeSize()));
    t.set_total_beta(static_cast<uint32_t>(gbShape.GetShapeSize()));
    t.set_total_y(static_cast<uint32_t>(static_cast<uint64_t>(M) * static_cast<uint64_t>(N)));

    // One task per block; cap to a reasonable number of blocks.
    uint32_t blockDim = static_cast<uint32_t>(totalTasks64);
    // Conservative cap to avoid oversubscription; runtime will schedule.
    if (blockDim > 256u) blockDim = 256u;
    if (blockDim == 0) blockDim = 1;

    t.set_block_dim(blockDim);
    context->SetBlockDim(blockDim);

    t.SaveToBuffer(context->GetRawTilingData()->GetData(),
                   context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(t.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class GemmBiasAddHardtanhMishGroupNormCustom : public OpDef {
public:
    explicit GemmBiasAddHardtanhMishGroupNormCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("lin_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gn_gamma").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gn_beta").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GemmBiasAddHardtanhMishGroupNormCustom);

} // namespace ops
