
#include "emsa_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace optiling {

// Kernel caps must match kernel_src.
static constexpr uint32_t MAX_NQ = 64;
static constexpr uint32_t MAX_NK = 64;
static constexpr uint32_t MAX_DK = 64;
static constexpr uint32_t MAX_DV = 64;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    auto qShape = context->GetInputShape(0)->GetStorageShape();
    auto kShape = context->GetInputShape(1)->GetStorageShape();
    auto vShape = context->GetInputShape(2)->GetStorageShape();

    if (qShape.GetDimNum() != 4 || kShape.GetDimNum() != 4 || vShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t B  = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t H  = static_cast<uint32_t>(qShape.GetDim(1));
    const uint32_t NQ = static_cast<uint32_t>(qShape.GetDim(2));
    const uint32_t DK = static_cast<uint32_t>(qShape.GetDim(3));

    const uint32_t Bk  = static_cast<uint32_t>(kShape.GetDim(0));
    const uint32_t Hk  = static_cast<uint32_t>(kShape.GetDim(1));
    const uint32_t DKk = static_cast<uint32_t>(kShape.GetDim(2));
    const uint32_t NK  = static_cast<uint32_t>(kShape.GetDim(3));

    const uint32_t Bv  = static_cast<uint32_t>(vShape.GetDim(0));
    const uint32_t Hv  = static_cast<uint32_t>(vShape.GetDim(1));
    const uint32_t NKv = static_cast<uint32_t>(vShape.GetDim(2));
    const uint32_t DV  = static_cast<uint32_t>(vShape.GetDim(3));

    if (B == 0 || H == 0 || NQ == 0 || DK == 0 || NK == 0 || DV == 0) return ge::GRAPH_FAILED;
    if (Bk != B || Hk != H || DKk != DK) return ge::GRAPH_FAILED;
    if (Bv != B || Hv != H || NKv != NK) return ge::GRAPH_FAILED;

    if (NQ > MAX_NQ || NK > MAX_NK || DK > MAX_DK || DV > MAX_DV) return ge::GRAPH_FAILED;

    EMSACustomTilingData tiling;
    tiling.set_B(B);
    tiling.set_H(H);
    tiling.set_NQ(NQ);
    tiling.set_DK(DK);
    tiling.set_NK(NK);
    tiling.set_DV(DV);
    tiling.set_scale(1.0f / std::sqrt(static_cast<float>(DK)));

    const uint32_t totalBH = B * H;
    tiling.set_totalBH(totalBH);

    // One block per (b,h) to maximize K/V reuse.
    uint32_t blockDim = std::max<uint32_t>(1, totalBH);
    context->SetBlockDim(blockDim);

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
    const auto* qShape = context->GetInputShape(0);
    const auto* vShape = context->GetInputShape(2);
    auto* outShape = context->GetOutputShape(0);
    if (qShape == nullptr || vShape == nullptr || outShape == nullptr) return GRAPH_FAILED;

    outShape->SetDimNum(4);
    outShape->SetDim(0, qShape->GetDim(0));
    outShape->SetDim(1, qShape->GetDim(1));
    outShape->SetDim(2, qShape->GetDim(2));
    outShape->SetDim(3, vShape->GetDim(3));
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
class EMSACustom : public OpDef {
public:
    explicit EMSACustom(const char* name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v")
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

OP_ADD(EMSACustom);
} // namespace ops
