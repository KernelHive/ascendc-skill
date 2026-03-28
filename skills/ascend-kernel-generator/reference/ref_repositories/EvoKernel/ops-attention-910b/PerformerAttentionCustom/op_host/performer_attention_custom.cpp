
#include "performer_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace optiling {

static inline uint32_t ClampU32(uint32_t v, uint32_t lo, uint32_t hi)
{
    return std::min<uint32_t>(hi, std::max<uint32_t>(lo, v));
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    // q_phi/k_phi: [B,H,S,F], v: [B,H,S,D]
    auto qShape = context->GetInputShape(0)->GetStorageShape();
    auto kShape = context->GetInputShape(1)->GetStorageShape();
    auto vShape = context->GetInputShape(2)->GetStorageShape();

    if (qShape.GetDimNum() != 4 || kShape.GetDimNum() != 4 || vShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t b = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t h = static_cast<uint32_t>(qShape.GetDim(1));
    const uint32_t s = static_cast<uint32_t>(qShape.GetDim(2));
    const uint32_t f = static_cast<uint32_t>(qShape.GetDim(3));
    if (b == 0 || h == 0 || s == 0 || f == 0) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(kShape.GetDim(0)) != b ||
        static_cast<uint32_t>(kShape.GetDim(1)) != h ||
        static_cast<uint32_t>(kShape.GetDim(2)) != s ||
        static_cast<uint32_t>(kShape.GetDim(3)) != f) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t vb = static_cast<uint32_t>(vShape.GetDim(0));
    const uint32_t vh = static_cast<uint32_t>(vShape.GetDim(1));
    const uint32_t vs = static_cast<uint32_t>(vShape.GetDim(2));
    const uint32_t d  = static_cast<uint32_t>(vShape.GetDim(3));
    if (vb != b || vh != h || vs != s || d == 0) return ge::GRAPH_FAILED;

    if (context->GetInputTensor(0)->GetDataType() != ge::DT_FLOAT ||
        context->GetInputTensor(1)->GetDataType() != ge::DT_FLOAT ||
        context->GetInputTensor(2)->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    // Kernel caps (match kernel constants)
    if (s > 4096u) return ge::GRAPH_FAILED;
    if (f > 256u)  return ge::GRAPH_FAILED;
    if (d > 128u)  return ge::GRAPH_FAILED;

    PerformerAttentionCustomTilingData tiling;
    tiling.set_b(b);
    tiling.set_h(h);
    tiling.set_s(s);
    tiling.set_f(f);
    tiling.set_d(d);
    tiling.set_eps(1e-6f);

    const uint64_t block_dim64 = static_cast<uint64_t>(b) * h;
    const uint32_t block_dim = static_cast<uint32_t>(std::min<uint64_t>(block_dim64, 65535ull));
    if (block_dim == 0) return ge::GRAPH_FAILED;
    context->SetBlockDim(block_dim);
    tiling.set_block_dim(block_dim);

    uint32_t d_tile = 64u;
    if (d < d_tile) d_tile = d;
    d_tile = ClampU32(d_tile, 1u, 64u);
    tiling.set_d_tile(d_tile);

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
    auto* yShape = context->GetOutputShape(0);
    if (qShape == nullptr || vShape == nullptr || yShape == nullptr) return GRAPH_FAILED;

    *yShape = *qShape;                   // [B,H,S,F]
    yShape->SetDim(3, vShape->GetDim(3)); // -> [B,H,S,D]
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
class PerformerAttentionCustom : public OpDef {
public:
    explicit PerformerAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("q_phi")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k_phi")
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

OP_ADD(PerformerAttentionCustom);
} // namespace ops
