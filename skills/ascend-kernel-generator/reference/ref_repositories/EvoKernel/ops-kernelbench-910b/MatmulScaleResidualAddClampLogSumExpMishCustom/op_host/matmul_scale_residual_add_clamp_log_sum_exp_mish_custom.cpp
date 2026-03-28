
#include "matmul_scale_residual_add_clamp_log_sum_exp_mish_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Specialized contract (must match python binding + kernel)
static constexpr uint32_t M_EXPECT = 1024;
static constexpr uint32_t K_EXPECT = 8192;
static constexpr uint32_t N_EXPECT = 8192;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    MatmulScaleResidualAddClampLogSumExpMishCustomTilingData tiling;

    const auto xShape = context->GetInputShape(0)->GetStorageShape();
    const auto wShape = context->GetInputShape(1)->GetStorageShape();
    const auto bShape = context->GetInputShape(2)->GetStorageShape();
    const auto sShape = context->GetInputShape(3)->GetStorageShape();
    const auto mnShape = context->GetInputShape(4)->GetStorageShape();
    const auto mxShape = context->GetInputShape(5)->GetStorageShape();

    if (xShape.GetDimNum() != 2 || wShape.GetDimNum() != 2) return ge::GRAPH_FAILED;
    if (bShape.GetDimNum() != 1) return ge::GRAPH_FAILED;
    if (sShape.GetDimNum() != 1 || mnShape.GetDimNum() != 1 || mxShape.GetDimNum() != 1) return ge::GRAPH_FAILED;

    const uint32_t M = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t K = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t N = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t wK = static_cast<uint32_t>(wShape.GetDim(1));

    if (M != M_EXPECT || K != K_EXPECT || N != N_EXPECT || wK != K_EXPECT) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(bShape.GetDim(0)) != N_EXPECT) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(sShape.GetDim(0)) != 1u) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(mnShape.GetDim(0)) != 1u) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(mxShape.GetDim(0)) != 1u) return ge::GRAPH_FAILED;

    // Validate dtypes via input tensors (more robust across gert versions)
    for (int i = 0; i < 6; ++i) {
        if (context->GetInputTensor(i) == nullptr) return ge::GRAPH_FAILED;
        if (context->GetInputTensor(i)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    }

    tiling.set_M(M);
    tiling.set_K(K);
    tiling.set_N(N);

    const uint64_t totalX = static_cast<uint64_t>(M) * static_cast<uint64_t>(K);
    const uint64_t totalW = static_cast<uint64_t>(N) * static_cast<uint64_t>(K);
    const uint64_t totalB = static_cast<uint64_t>(N);
    const uint64_t totalY = static_cast<uint64_t>(M); // output is [M,1]

    if (totalX > 0xFFFFFFFFull || totalW > 0xFFFFFFFFull || totalY > 0xFFFFFFFFull) return ge::GRAPH_FAILED;

    tiling.set_total_x(static_cast<uint32_t>(totalX));
    tiling.set_total_w(static_cast<uint32_t>(totalW));
    tiling.set_total_b(static_cast<uint32_t>(totalB));
    tiling.set_total_y(static_cast<uint32_t>(totalY));

    // Parallelize over rows
    uint32_t blockDim = 48;
    if (blockDim > M) blockDim = M;
    if (blockDim == 0) blockDim = 1;
    const uint32_t rowsPerBlock = (M + blockDim - 1) / blockDim;

    tiling.set_block_dim(blockDim);
    tiling.set_rows_per_block(rowsPerBlock);

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

class MatmulScaleResidualAddClampLogSumExpMishCustom : public OpDef {
public:
    explicit MatmulScaleResidualAddClampLogSumExpMishCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scaling").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("clamp_min").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("clamp_max").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MatmulScaleResidualAddClampLogSumExpMishCustom);

} // namespace ops
