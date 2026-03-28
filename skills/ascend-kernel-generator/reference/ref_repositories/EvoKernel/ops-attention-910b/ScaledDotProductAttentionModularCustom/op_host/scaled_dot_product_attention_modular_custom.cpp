
#include "scaled_dot_product_attention_modular_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    auto qShape = context->GetInputShape(0)->GetStorageShape();
    auto kShape = context->GetInputShape(1)->GetStorageShape();
    auto vShape = context->GetInputShape(2)->GetStorageShape();

    if (qShape.GetDimNum() != 4 || kShape.GetDimNum() != 4 || vShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t b  = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t h  = static_cast<uint32_t>(qShape.GetDim(1));
    const uint32_t nq = static_cast<uint32_t>(qShape.GetDim(2));
    const uint32_t dk = static_cast<uint32_t>(qShape.GetDim(3));
    if (b == 0 || h == 0 || nq == 0 || dk == 0) return ge::GRAPH_FAILED;

    const uint32_t kb  = static_cast<uint32_t>(kShape.GetDim(0));
    const uint32_t kh  = static_cast<uint32_t>(kShape.GetDim(1));
    const uint32_t nk  = static_cast<uint32_t>(kShape.GetDim(2));
    const uint32_t kdk = static_cast<uint32_t>(kShape.GetDim(3));
    if (kb != b || kh != h || nk == 0 || kdk != dk) return ge::GRAPH_FAILED;

    const uint32_t vb  = static_cast<uint32_t>(vShape.GetDim(0));
    const uint32_t vh  = static_cast<uint32_t>(vShape.GetDim(1));
    const uint32_t vnk = static_cast<uint32_t>(vShape.GetDim(2));
    const uint32_t dv  = static_cast<uint32_t>(vShape.GetDim(3));
    if (vb != b || vh != h || vnk != nk || dv == 0) return ge::GRAPH_FAILED;

    if (context->GetInputTensor(0)->GetDataType() != ge::DT_FLOAT ||
        context->GetInputTensor(1)->GetDataType() != ge::DT_FLOAT ||
        context->GetInputTensor(2)->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    ScaledDotProductAttentionModularCustomTilingData tiling;
    tiling.set_b(b);
    tiling.set_h(h);
    tiling.set_nq(nq);
    tiling.set_nk(nk);
    tiling.set_dk(dk);
    tiling.set_dv(dv);
    tiling.set_scale(1.0f / std::sqrt(static_cast<float>(dk)));

    const uint64_t total_rows = static_cast<uint64_t>(b) * h * nq;

    // Avoid absurdly large blockDim when each block has substantial serial work (nk loop).
    // A conservative cap that usually maps well to available AICore count.
    constexpr uint32_t CAP = 1024;
    uint32_t block_dim = static_cast<uint32_t>(std::min<uint64_t>(total_rows, static_cast<uint64_t>(CAP)));
    block_dim = std::max(block_dim, 1u);
    context->SetBlockDim(block_dim);
    tiling.set_block_dim(block_dim);

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

    *outShape = *qShape;                    // [B,H,NQ,Dk]
    outShape->SetDim(3, vShape->GetDim(3)); // -> [B,H,NQ,Dv]
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
class ScaledDotProductAttentionModularCustom : public OpDef {
public:
    explicit ScaledDotProductAttentionModularCustom(const char* name) : OpDef(name)
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

        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ScaledDotProductAttentionModularCustom);
} // namespace ops
