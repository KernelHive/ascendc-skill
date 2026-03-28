
#include "gated_channel_transform_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static inline uint32_t AlignUp(uint32_t v, uint32_t a) {
    return (v + a - 1U) / a * a;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GatedChannelTransformCustomTilingData tiling;

    const auto* xShapePtr = context->GetInputShape(0);
    const auto* aShapePtr = context->GetInputShape(1);
    const auto* gShapePtr = context->GetInputShape(2);
    const auto* bShapePtr = context->GetInputShape(3);
    const auto* eShapePtr = context->GetInputShape(4);
    if (xShapePtr == nullptr || aShapePtr == nullptr || gShapePtr == nullptr ||
        bShapePtr == nullptr || eShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& xs = xShapePtr->GetStorageShape();
    const auto& as = aShapePtr->GetStorageShape();
    const auto& gs = gShapePtr->GetStorageShape();
    const auto& bs = bShapePtr->GetStorageShape();
    const auto& es = eShapePtr->GetStorageShape();

    // x: [N,C,H,W], alpha/gamma/beta: [C], epsilon: scalar (1 element)
    if (xs.GetDimNum() != 4) return ge::GRAPH_FAILED;
    if (as.GetDimNum() != 1) return ge::GRAPH_FAILED;
    if (gs.GetDimNum() != 1) return ge::GRAPH_FAILED;
    if (bs.GetDimNum() != 1) return ge::GRAPH_FAILED;
    if (es.GetShapeSize() != 1) return ge::GRAPH_FAILED;

    uint32_t N = static_cast<uint32_t>(xs.GetDim(0));
    uint32_t C = static_cast<uint32_t>(xs.GetDim(1));
    uint32_t H = static_cast<uint32_t>(xs.GetDim(2));
    uint32_t W = static_cast<uint32_t>(xs.GetDim(3));
    if (N == 0U || C == 0U || H == 0U || W == 0U) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(as.GetDim(0)) != C) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(gs.GetDim(0)) != C) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(bs.GetDim(0)) != C) return ge::GRAPH_FAILED;

    uint32_t HW = H * W;
    if (HW == 0U) return ge::GRAPH_FAILED;

    uint32_t CHW = C * HW;
    if (CHW == 0U) return ge::GRAPH_FAILED;

    uint32_t totalLength = N * CHW;
    if (totalLength == 0U) return ge::GRAPH_FAILED;

    // 32B alignment => 8 floats
    uint32_t CAlign = AlignUp(C, 8U);
    if (CAlign == 0U) CAlign = 8U;

    tiling.set_N(N);
    tiling.set_C(C);
    tiling.set_H(H);
    tiling.set_W(W);
    tiling.set_HW(HW);
    tiling.set_CHW(CHW);
    tiling.set_totalLength(totalLength);
    tiling.set_CAlign(CAlign);
    tiling.set_invC(1.0f / static_cast<float>(C));

    // temp buffer for tanh
    tiling.set_tanhTmpBytes(16U * 1024U);

    // parallelize over batch
    uint32_t block_dim = std::min<uint32_t>(N, 32U);
    if (block_dim == 0U) block_dim = 1U;
    context->SetBlockDim(block_dim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
class GatedChannelTransformCustom : public OpDef {
public:
    explicit GatedChannelTransformCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("alpha")
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
        this->Input("epsilon")
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

OP_ADD(GatedChannelTransformCustom);
} // namespace ops
