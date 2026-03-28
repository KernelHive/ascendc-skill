
#include "gemm_multiply_leaky_relu_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Fixed specialized contract for performance & simplicity:
// x: [1024,8192] float32 contiguous
// w: [8192,8192] float32 contiguous (PyTorch Linear weight: [out,in] == [N,K])
// b: [8192] float32 contiguous
// multiplier: [1] float32 contiguous (scalar tensor)
// negative_slope: [1] float32 contiguous (scalar tensor)
// y: [1024,8192] float32
static constexpr uint32_t M = 1024;
static constexpr uint32_t K = 8192;
static constexpr uint32_t N = 8192;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto sx = context->GetInputShape(0);
    auto sw = context->GetInputShape(1);
    auto sb = context->GetInputShape(2);
    auto sm = context->GetInputShape(3);
    auto ss = context->GetInputShape(4);
    if (sx == nullptr || sw == nullptr || sb == nullptr || sm == nullptr || ss == nullptr ||
        context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& x = sx->GetOriginShape();
    const auto& w = sw->GetOriginShape();
    const auto& b = sb->GetOriginShape();
    const auto& mul = sm->GetOriginShape();
    const auto& ns = ss->GetOriginShape();

    if (x.GetDimNum() != 2 || w.GetDimNum() != 2 || b.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    // scalar tensors can be [] or [1] depending on framework path; accept both.
    const bool mulOk = (mul.GetDimNum() == 0) || (mul.GetDimNum() == 1 && (uint32_t)mul.GetDim(0) == 1);
    const bool nsOk  = (ns.GetDimNum() == 0)  || (ns.GetDimNum() == 1 && (uint32_t)ns.GetDim(0) == 1);
    if (!mulOk || !nsOk) {
        return ge::GRAPH_FAILED;
    }

    if ((uint32_t)x.GetDim(0) != M || (uint32_t)x.GetDim(1) != K) return ge::GRAPH_FAILED;
    if ((uint32_t)w.GetDim(0) != N || (uint32_t)w.GetDim(1) != K) return ge::GRAPH_FAILED;
    if ((uint32_t)b.GetDim(0) != N) return ge::GRAPH_FAILED;

    if (context->GetInputTensor(0)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(1)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(2)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(3)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(4)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;

    GemmMultiplyLeakyReluCustomTilingData tiling;
    tiling.set_totalX(sx->GetStorageShape().GetShapeSize());
    tiling.set_totalW(sw->GetStorageShape().GetShapeSize());
    tiling.set_totalB(sb->GetStorageShape().GetShapeSize());
    tiling.set_totalMultiplier(sm->GetStorageShape().GetShapeSize());
    tiling.set_totalNegativeSlope(ss->GetStorageShape().GetShapeSize());
    tiling.set_totalY(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());

    tiling.set_M(M);
    tiling.set_K(K);
    tiling.set_N(N);
    tiling.set_totalElems(M * N);

    // Parallelize over output elements (flattened M*N) so all elements are covered.
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

class GemmMultiplyLeakyReluCustom : public OpDef {
public:
    explicit GemmMultiplyLeakyReluCustom(const char* name) : OpDef(name)
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

        this->Input("multiplier")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("negative_slope")
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

OP_ADD(GemmMultiplyLeakyReluCustom);

} // namespace ops
