
#include "gemm_scaling_hardtanh_gelu_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Fixed specialized contract for performance & simplicity:
// x: [2048,8192] float32 contiguous
// w: [8192,8192] float32 contiguous (PyTorch Linear weight: [out,in] == [N,K])
// b: [8192] float32 contiguous
// scaling/min/max: [1] float32 contiguous
// y: [2048,8192] float32
static constexpr uint32_t M = 2048;
static constexpr uint32_t K = 8192;
static constexpr uint32_t N = 8192;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto sx = context->GetInputShape(0);
    auto sw = context->GetInputShape(1);
    auto sb = context->GetInputShape(2);
    auto ss = context->GetInputShape(3);
    auto smin = context->GetInputShape(4);
    auto smax = context->GetInputShape(5);
    if (sx == nullptr || sw == nullptr || sb == nullptr || ss == nullptr ||
        smin == nullptr || smax == nullptr || context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& x = sx->GetOriginShape();
    const auto& w = sw->GetOriginShape();
    const auto& b = sb->GetOriginShape();
    const auto& s = ss->GetOriginShape();
    const auto& mn = smin->GetOriginShape();
    const auto& mx = smax->GetOriginShape();

    if (x.GetDimNum() != 2 || w.GetDimNum() != 2 || b.GetDimNum() != 1 ||
        s.GetDimNum() != 1 || mn.GetDimNum() != 1 || mx.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    if ((uint32_t)x.GetDim(0) != M || (uint32_t)x.GetDim(1) != K) return ge::GRAPH_FAILED;
    if ((uint32_t)w.GetDim(0) != N || (uint32_t)w.GetDim(1) != K) return ge::GRAPH_FAILED;
    if ((uint32_t)b.GetDim(0) != N) return ge::GRAPH_FAILED;
    if ((uint32_t)s.GetDim(0) != 1) return ge::GRAPH_FAILED;
    if ((uint32_t)mn.GetDim(0) != 1) return ge::GRAPH_FAILED;
    if ((uint32_t)mx.GetDim(0) != 1) return ge::GRAPH_FAILED;

    if (context->GetInputTensor(0)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(1)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(2)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(3)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(4)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (context->GetInputTensor(5)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;

    GemmScalingHardtanhGeluCustomTilingData tiling;
    tiling.set_totalX(sx->GetStorageShape().GetShapeSize());
    tiling.set_totalW(sw->GetStorageShape().GetShapeSize());
    tiling.set_totalB(sb->GetStorageShape().GetShapeSize());
    tiling.set_totalS(ss->GetStorageShape().GetShapeSize());
    tiling.set_totalMin(smin->GetStorageShape().GetShapeSize());
    tiling.set_totalMax(smax->GetStorageShape().GetShapeSize());
    tiling.set_totalY(context->GetOutputShape(0)->GetStorageShape().GetShapeSize());

    tiling.set_M(M);
    tiling.set_K(K);
    tiling.set_N(N);
    tiling.set_totalElems(M * N);

    // UB tile for post-ops (scale+clamp+erf gelu). Conservative to keep temp buffers small.
    tiling.set_tileElems(1024);

    // Parallelize over flattened output elements.
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

class GemmScalingHardtanhGeluCustom : public OpDef {
public:
    explicit GemmScalingHardtanhGeluCustom(const char* name) : OpDef(name)
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

        this->Input("scaling")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("hardtanh_min")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("hardtanh_max")
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

OP_ADD(GemmScalingHardtanhGeluCustom);

} // namespace ops
