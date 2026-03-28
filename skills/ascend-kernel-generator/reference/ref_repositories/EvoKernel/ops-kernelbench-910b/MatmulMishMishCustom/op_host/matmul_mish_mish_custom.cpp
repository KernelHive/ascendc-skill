
#include "matmul_mish_mish_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static constexpr uint32_t M_EXPECT = 1024;
static constexpr uint32_t K_EXPECT = 8192;
static constexpr uint32_t N_EXPECT = 8192;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MatmulMishMishCustomTilingData tiling;

    const auto xShape = context->GetInputShape(0)->GetStorageShape();
    const auto wShape = context->GetInputShape(1)->GetStorageShape();
    const auto bShape = context->GetInputShape(2)->GetStorageShape();

    if (xShape.GetDimNum() != 2 || wShape.GetDimNum() != 2) return ge::GRAPH_FAILED;
    if (bShape.GetDimNum() != 1) return ge::GRAPH_FAILED;

    const uint32_t M = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t K = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t N = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t wK = static_cast<uint32_t>(wShape.GetDim(1));

    if (M != M_EXPECT || K != K_EXPECT || N != N_EXPECT || wK != K_EXPECT) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(bShape.GetDim(0)) != N_EXPECT) return ge::GRAPH_FAILED;

    tiling.set_M(M);
    tiling.set_K(K);
    tiling.set_N(N);

    const uint64_t totalY = static_cast<uint64_t>(M) * static_cast<uint64_t>(N);
    if (totalY == 0 || totalY > 0xFFFFFFFFull) return ge::GRAPH_FAILED;
    tiling.set_total_elems(static_cast<uint32_t>(totalY));

    // Moderate blockDim to avoid launch overhead; each block handles a chunk of output elements.
    uint32_t blockDim = 96;
    if (blockDim > static_cast<uint32_t>(totalY)) blockDim = static_cast<uint32_t>(totalY);
    if (blockDim == 0) blockDim = 1;

    const uint32_t elemsPerBlock = (static_cast<uint32_t>(totalY) + blockDim - 1) / blockDim;

    tiling.set_block_dim(blockDim);
    tiling.set_elems_per_block(elemsPerBlock);

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

class MatmulMishMishCustom : public OpDef {
public:
    explicit MatmulMishMishCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("bias")
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

OP_ADD(MatmulMishMishCustom);

} // namespace ops
