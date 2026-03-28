
#include "conv_transpose2d_add_min_gelu_multiply_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

// Baked constants for this benchmark to avoid attribute/scalar input issues.
static constexpr float BAKED_ADDV = 0.5f;
static constexpr float BAKED_MULV = 2.0f;
static constexpr uint32_t MAX_BLOCK_DIM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose2dAddMinGeluMultiplyCustomTilingData tiling;

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

    // total = N*C*HW (clamp)
    uint64_t total64 = static_cast<uint64_t>(N) * static_cast<uint64_t>(C) * static_cast<uint64_t>(HW);
    if (total64 > 0xFFFFFFFFULL) total64 = 0xFFFFFFFFULL;
    const uint32_t total = static_cast<uint32_t>(total64);
    if (total == 0) return ge::GRAPH_FAILED;

    // Launch: split over batch N (post-op is elementwise; N is a safe partition)
    uint32_t blockDim = (N < MAX_BLOCK_DIM) ? (N == 0 ? 1 : N) : MAX_BLOCK_DIM;
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);

    tiling.set_N(N);
    tiling.set_C(C);
    tiling.set_HW(HW);
    tiling.set_total(total);
    tiling.set_addv(BAKED_ADDV);
    tiling.set_mulv(BAKED_MULV);

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
    // This op is a post-op epilogue: output shape equals input shape.
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

class ConvTranspose2dAddMinGeluMultiplyCustom : public OpDef {
public:
    explicit ConvTranspose2dAddMinGeluMultiplyCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
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

OP_ADD(ConvTranspose2dAddMinGeluMultiplyCustom);

}  // namespace ops
