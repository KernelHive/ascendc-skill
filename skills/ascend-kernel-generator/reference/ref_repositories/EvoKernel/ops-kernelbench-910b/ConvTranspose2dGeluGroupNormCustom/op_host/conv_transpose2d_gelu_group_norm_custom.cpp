
#include "conv_transpose2d_gelu_group_norm_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Bake to keep a stable interface and avoid attr wiring pitfalls.
static constexpr uint32_t BAKED_G = 8;
static constexpr float BAKED_EPS = 1e-5f;
static constexpr uint32_t MAX_BLOCK_DIM = 64;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose2dGeluGroupNormCustomTilingData tiling;

    const auto* xShape = context->GetInputShape(0);
    const auto& s = xShape->GetOriginShape();
    const size_t rank = s.GetDimNum();
    if (rank < 2) return ge::GRAPH_FAILED;

    const int64_t N64 = s.GetDim(0);
    const int64_t C64 = s.GetDim(1);
    if (N64 <= 0 || C64 <= 0) return ge::GRAPH_FAILED;

    uint64_t HW64 = 1;
    for (size_t i = 2; i < rank; ++i) {
        const int64_t d = s.GetDim(i);
        if (d <= 0) return ge::GRAPH_FAILED;
        HW64 *= static_cast<uint64_t>(d);
        if (HW64 > 0xFFFFFFFFULL) { HW64 = 0xFFFFFFFFULL; break; }
    }
    if (HW64 == 0) return ge::GRAPH_FAILED;

    const uint32_t N = static_cast<uint32_t>(N64 > 0xFFFFFFFFLL ? 0xFFFFFFFFu : static_cast<uint32_t>(N64));
    const uint32_t C = static_cast<uint32_t>(C64 > 0xFFFFFFFFLL ? 0xFFFFFFFFu : static_cast<uint32_t>(C64));
    const uint32_t HW = static_cast<uint32_t>(HW64);

    const uint32_t G = BAKED_G;
    if (G == 0 || (C % G) != 0) return ge::GRAPH_FAILED;
    const uint32_t groupSize = C / G;
    if (groupSize == 0) return ge::GRAPH_FAILED;

    uint64_t groupsTotal64 = static_cast<uint64_t>(N) * static_cast<uint64_t>(G);
    if (groupsTotal64 == 0) return ge::GRAPH_FAILED;
    if (groupsTotal64 > 0xFFFFFFFFULL) groupsTotal64 = 0xFFFFFFFFULL;
    const uint32_t groupsTotal = static_cast<uint32_t>(groupsTotal64);

    uint32_t blockDim = (groupsTotal < MAX_BLOCK_DIM) ? (groupsTotal == 0 ? 1 : groupsTotal) : MAX_BLOCK_DIM;
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);

    const uint64_t reduceCount64 = static_cast<uint64_t>(groupSize) * static_cast<uint64_t>(HW);
    if (reduceCount64 == 0) return ge::GRAPH_FAILED;
    const float invReduce = 1.0f / static_cast<float>(reduceCount64);

    tiling.set_N(N);
    tiling.set_C(C);
    tiling.set_HW(HW);
    tiling.set_G(G);
    tiling.set_groupSize(groupSize);
    tiling.set_groupsTotal(groupsTotal);
    tiling.set_invReduce(invReduce);
    tiling.set_eps(BAKED_EPS);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    // This op is an epilogue post-op (run after ConvTranspose2d): output shape equals input shape.
    const gert::Shape* x = context->GetInputShape(0);
    gert::Shape* y = context->GetOutputShape(0);
    *y = *x;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class ConvTranspose2dGeluGroupNormCustom : public OpDef {
public:
    explicit ConvTranspose2dGeluGroupNormCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose2dGeluGroupNormCustom);

}  // namespace ops
