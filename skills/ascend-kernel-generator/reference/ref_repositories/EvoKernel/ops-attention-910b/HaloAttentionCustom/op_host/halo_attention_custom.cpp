
#include "halo_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    // q:[B,I,D], k/v:[B,J,D], mask:[B,1,J] bool (True => masked)
    auto qShape = context->GetInputShape(0)->GetStorageShape();
    auto kShape = context->GetInputShape(1)->GetStorageShape();
    auto vShape = context->GetInputShape(2)->GetStorageShape();
    auto mShape = context->GetInputShape(3)->GetStorageShape();

    if (qShape.GetDimNum() != 3 || kShape.GetDimNum() != 3 || vShape.GetDimNum() != 3 || mShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t B = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t I = static_cast<uint32_t>(qShape.GetDim(1));
    const uint32_t D = static_cast<uint32_t>(qShape.GetDim(2));

    const uint32_t kB = static_cast<uint32_t>(kShape.GetDim(0));
    const uint32_t J  = static_cast<uint32_t>(kShape.GetDim(1));
    const uint32_t kD = static_cast<uint32_t>(kShape.GetDim(2));

    if (B == 0 || I == 0 || J == 0 || D == 0) return ge::GRAPH_FAILED;
    if (kB != B || kD != D) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(vShape.GetDim(0)) != B ||
        static_cast<uint32_t>(vShape.GetDim(1)) != J ||
        static_cast<uint32_t>(vShape.GetDim(2)) != D) {
        return ge::GRAPH_FAILED;
    }

    if (static_cast<uint32_t>(mShape.GetDim(0)) != B ||
        static_cast<uint32_t>(mShape.GetDim(1)) != 1U ||
        static_cast<uint32_t>(mShape.GetDim(2)) != J) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t totalRows = B * I;

    // Conservative core count for stability; spread rows across cores
    uint32_t coreNum = std::min<uint32_t>(std::max<uint32_t>(1U, totalRows), 24U);
    context->SetBlockDim(coreNum);

    HaloAttentionCustomTilingData tiling;
    tiling.set_B(B);
    tiling.set_I(I);
    tiling.set_J(J);
    tiling.set_D(D);
    tiling.set_totalRows(totalRows);
    tiling.set_coreNum(coreNum);

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
    auto* outShape = context->GetOutputShape(0);
    const auto* qShape = context->GetInputShape(0);
    if (outShape == nullptr || qShape == nullptr) return GRAPH_FAILED;
    *outShape = *qShape; // [B,I,D]
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class HaloAttentionCustom : public OpDef {
public:
    explicit HaloAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mask").ParamType(REQUIRED).DataType({ge::DT_BOOL}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(HaloAttentionCustom);
} // namespace ops
