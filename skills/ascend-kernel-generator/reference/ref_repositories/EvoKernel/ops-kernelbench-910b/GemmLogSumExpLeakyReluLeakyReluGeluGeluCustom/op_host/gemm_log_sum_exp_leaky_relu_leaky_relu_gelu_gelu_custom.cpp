
#include "gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Fixed specialized contract for performance & simplicity:
// x: [1024,8192] float32 contiguous
// w: [8192,8192] float32 contiguous (PyTorch Linear weight: [out,in] == [N,K])
// b: [8192]      float32 contiguous (bias required)
// y: [1024,1]    float32 contiguous
static constexpr uint32_t M = 1024;
static constexpr uint32_t K = 8192;
static constexpr uint32_t N = 8192;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto sx = context->GetInputShape(0);
    auto sw = context->GetInputShape(1);
    auto sb = context->GetInputShape(2);
    auto sy = context->GetOutputShape(0);
    if (sx == nullptr || sw == nullptr || sb == nullptr || sy == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& x = sx->GetOriginShape();
    const auto& w = sw->GetOriginShape();
    const auto& b = sb->GetOriginShape();
    const auto& y = sy->GetOriginShape();

    if (x.GetDimNum() != 2 || w.GetDimNum() != 2 || b.GetDimNum() != 1 || y.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    if ((uint32_t)x.GetDim(0) != M || (uint32_t)x.GetDim(1) != K) return ge::GRAPH_FAILED;
    if ((uint32_t)w.GetDim(0) != N || (uint32_t)w.GetDim(1) != K) return ge::GRAPH_FAILED;
    if ((uint32_t)b.GetDim(0) != N) return ge::GRAPH_FAILED;

    // Output is fixed [M,1]
    if ((uint32_t)y.GetDim(0) != M || (uint32_t)y.GetDim(1) != 1u) return ge::GRAPH_FAILED;

    if (context->GetInputTensor(0) == nullptr || context->GetInputTensor(1) == nullptr ||
        context->GetInputTensor(2) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputTensor(0)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(1)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(2)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;

    GemmLogSumExpLeakyReluLeakyReluGeluGeluCustomTilingData tiling;
    tiling.set_totalX(sx->GetStorageShape().GetShapeSize());
    tiling.set_totalW(sw->GetStorageShape().GetShapeSize());
    tiling.set_totalB(sb->GetStorageShape().GetShapeSize());
    tiling.set_totalY(sy->GetStorageShape().GetShapeSize());

    tiling.set_M(M);
    tiling.set_K(K);
    tiling.set_N(N);

    // Parallelize over rows; each core handles a chunk of rows.
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

class GemmLogSumExpLeakyReluLeakyReluGeluGeluCustom : public OpDef {
public:
    explicit GemmLogSumExpLeakyReluLeakyReluGeluGeluCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("w")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GemmLogSumExpLeakyReluLeakyReluGeluGeluCustom);

} // namespace ops
