
#include "un_permute_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static constexpr uint32_t MAX_TOPK = 8;
static constexpr uint32_t MAX_K    = 4096;
static constexpr uint32_t K_TILE   = 1024; // matches kernel UB plan

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    auto expShape  = context->GetInputShape(0)->GetStorageShape();
    auto topkShape = context->GetInputShape(1)->GetStorageShape();
    auto invShape  = context->GetInputShape(2)->GetStorageShape();

    if (expShape.GetDimNum() != 2 || topkShape.GetDimNum() != 2 || invShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t totalExp = static_cast<uint32_t>(expShape.GetDim(0));
    const uint32_t k        = static_cast<uint32_t>(expShape.GetDim(1));
    const uint32_t m        = static_cast<uint32_t>(topkShape.GetDim(0));
    const uint32_t topk     = static_cast<uint32_t>(topkShape.GetDim(1));
    const uint64_t invN     = static_cast<uint64_t>(invShape.GetDim(0));

    if (m == 0 || k == 0 || topk == 0 || totalExp == 0) return ge::GRAPH_FAILED;
    if (invN != static_cast<uint64_t>(m) * static_cast<uint64_t>(topk)) return ge::GRAPH_FAILED;

    if (topk > MAX_TOPK) return ge::GRAPH_FAILED;
    if (k > MAX_K) return ge::GRAPH_FAILED;

    UnPermuteCustomTilingData td;
    td.set_m(m);
    td.set_k(k);
    td.set_topk(topk);
    td.set_kTile(K_TILE);

    // Parallelize over token rows.
    uint32_t blockDim = m;
    if (blockDim == 0) blockDim = 1;
    if (blockDim > 48u) blockDim = 48u;
    context->SetBlockDim(blockDim);

    td.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(td.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) return GRAPH_FAILED;
    auto* outShape = context->GetOutputShape(0);
    const auto* expShape = context->GetInputShape(0);
    const auto* topkShape = context->GetInputShape(1);
    if (outShape == nullptr || expShape == nullptr || topkShape == nullptr) return GRAPH_FAILED;

    outShape->SetDimNum(2);
    outShape->SetDim(0, topkShape->GetDim(0));
    outShape->SetDim(1, expShape->GetDim(1));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) return GRAPH_FAILED;
    context->SetOutputDataType(0, context->GetInputDataType(0)); // bf16
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class UnPermuteCustom : public OpDef {
public:
    explicit UnPermuteCustom(const char* name) : OpDef(name)
    {
        this->Input("expert_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("topk_vals")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("inv_perm")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(UnPermuteCustom);
} // namespace ops
