
#include "eca_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace optiling {

static constexpr uint32_t MAX_K = 4096;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    auto xShape = context->GetInputShape(0)->GetStorageShape();
    auto wShape = context->GetInputShape(1)->GetStorageShape();
    if (xShape.GetDimNum() != 4 || wShape.GetDimNum() != 3) return ge::GRAPH_FAILED;

    const uint32_t B = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t C = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t H = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t W = static_cast<uint32_t>(xShape.GetDim(3));
    if (B == 0 || C == 0 || H == 0 || W == 0) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(wShape.GetDim(0)) != 1U) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(wShape.GetDim(1)) != 1U) return ge::GRAPH_FAILED;
    const uint32_t K = static_cast<uint32_t>(wShape.GetDim(2));
    if (K == 0 || K > MAX_K) return ge::GRAPH_FAILED;
    if ((K & 1U) == 0U) return ge::GRAPH_FAILED;

    const uint32_t HW = H * W;
    if (HW == 0) return ge::GRAPH_FAILED;

    ECAAttentionCustomTilingData tiling;
    tiling.set_B(B);
    tiling.set_C(C);
    tiling.set_H(H);
    tiling.set_W(W);
    tiling.set_HW(HW);
    tiling.set_K(K);
    tiling.set_pad((K - 1U) / 2U);
    tiling.set_invHW(1.0f / static_cast<float>(HW));

    // Conservative cTile to bound UB usage (pooled is full C).
    // Keep it relatively small to leave room for sigmoid tmp and other buffers.
    uint32_t cTile = 256;
    if (cTile > C) cTile = C;
    cTile = std::max<uint32_t>(1U, cTile);
    tiling.set_cTile(cTile);

    // Reserve explicit scratch for Sigmoid primitive.
    // 16KB is a common safe scratch size for small vector ops; keep as tiling param.
    tiling.set_sigTmpBytes(16U * 1024U);

    // Split over batch.
    // Prefer up to 32 blocks but never exceed B.
    uint32_t blockDim = std::min<uint32_t>(B, 32U);
    blockDim = std::max<uint32_t>(1U, blockDim);
    context->SetBlockDim(blockDim);

    tiling.set_totalB(B);
    tiling.set_bPerBlock((B + blockDim - 1U) / blockDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) return GRAPH_FAILED;
    const auto* xShape = context->GetInputShape(0);
    auto* outShape = context->GetOutputShape(0);
    if (xShape == nullptr || outShape == nullptr) return GRAPH_FAILED;

    // y has same shape as x
    outShape->SetDimNum(4);
    outShape->SetDim(0, xShape->GetDim(0));
    outShape->SetDim(1, xShape->GetDim(1));
    outShape->SetDim(2, xShape->GetDim(2));
    outShape->SetDim(3, xShape->GetDim(3));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) return GRAPH_FAILED;
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {
class ECAAttentionCustom : public OpDef {
public:
    explicit ECAAttentionCustom(const char* name) : OpDef(name)
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

OP_ADD(ECAAttentionCustom);
} // namespace ops
