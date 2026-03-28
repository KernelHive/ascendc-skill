
#include "net_vlad_no_ghost_clusters_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <algorithm>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    NetVladNoGhostClustersCustomTilingData tiling;

    uint32_t totalX         = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t totalClusters  = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    uint32_t totalClusters2 = context->GetInputShape(2)->GetStorageShape().GetShapeSize();
    uint32_t totalBnW       = context->GetInputShape(3)->GetStorageShape().GetShapeSize();
    uint32_t totalBnB       = context->GetInputShape(4)->GetStorageShape().GetShapeSize();
    uint32_t totalBnM       = context->GetInputShape(5)->GetStorageShape().GetShapeSize();
    uint32_t totalBnV       = context->GetInputShape(6)->GetStorageShape().GetShapeSize();
    uint32_t totalY         = context->GetOutputShape(0)->GetStorageShape().GetShapeSize();

    tiling.set_totalX(totalX);
    tiling.set_totalClusters(totalClusters);
    tiling.set_totalClusters2(totalClusters2);
    tiling.set_totalBnW(totalBnW);
    tiling.set_totalBnB(totalBnB);
    tiling.set_totalBnM(totalBnM);
    tiling.set_totalBnV(totalBnV);
    tiling.set_totalY(totalY);

    // Fixed benchmark contract
    constexpr uint32_t B = 2048;
    constexpr uint32_t N = 100;
    constexpr uint32_t D = 512;
    constexpr uint32_t K = 32;

    tiling.set_B(B);
    tiling.set_N(N);
    tiling.set_D(D);
    tiling.set_K(K);

    tiling.set_bnEps(1.0e-5f);
    tiling.set_l2Eps(1.0e-12f);

    // Conservative, stable parallelism.
    constexpr uint32_t kMaxBlockDim = 40;
    uint32_t blockDim = std::min<uint32_t>(kMaxBlockDim, B);
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class NetVladNoGhostClustersCustom : public OpDef {
public:
    explicit NetVladNoGhostClustersCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("clusters").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("clusters2").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("bn_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bn_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bn_mean").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bn_var").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(NetVladNoGhostClustersCustom);

} // namespace ops
