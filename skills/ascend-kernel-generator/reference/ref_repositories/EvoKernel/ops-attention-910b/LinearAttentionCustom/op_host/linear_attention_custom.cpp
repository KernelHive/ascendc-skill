
#include "linear_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
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

    const uint32_t b = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t h = static_cast<uint32_t>(qShape.GetDim(1));
    const uint32_t n = static_cast<uint32_t>(qShape.GetDim(2));
    const uint32_t d = static_cast<uint32_t>(qShape.GetDim(3));
    if (b == 0 || h == 0 || n == 0 || d == 0) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(kShape.GetDim(0)) != b || static_cast<uint32_t>(kShape.GetDim(1)) != h ||
        static_cast<uint32_t>(kShape.GetDim(2)) != n || static_cast<uint32_t>(kShape.GetDim(3)) != d) {
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint32_t>(vShape.GetDim(0)) != b || static_cast<uint32_t>(vShape.GetDim(1)) != h ||
        static_cast<uint32_t>(vShape.GetDim(2)) != n || static_cast<uint32_t>(vShape.GetDim(3)) != d) {
        return ge::GRAPH_FAILED;
    }

    if (context->GetInputTensor(0)->GetDataType() != ge::DT_FLOAT ||
        context->GetInputTensor(1)->GetDataType() != ge::DT_FLOAT ||
        context->GetInputTensor(2)->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    // Must match kernel caps and python checks.
    if (d > 64u) return ge::GRAPH_FAILED;
    if (n > 1024u) return ge::GRAPH_FAILED;

    const float eps = 1e-6f;

    LinearAttentionCustomTilingData tiling;
    tiling.set_b(b);
    tiling.set_h(h);
    tiling.set_n(n);
    tiling.set_d(d);
    tiling.set_eps(eps);

    const uint64_t total_bh_64 = static_cast<uint64_t>(b) * h;
    if (total_bh_64 == 0) return ge::GRAPH_FAILED;
    tiling.set_total_bh(static_cast<uint32_t>(std::min<uint64_t>(total_bh_64, 0xFFFFFFFFull)));

    // Split work across (bh, token rows) for better occupancy.
    // Each block handles row_block token rows for a given (b,h).
    // Choose a conservative row_block to increase parallelism without excessive overhead.
    uint32_t row_block = 8;
    if (n <= 64) row_block = 4;
    if (n <= 16) row_block = 2;
    tiling.set_row_block(row_block);

    const uint64_t total_tasks = total_bh_64 * ((static_cast<uint64_t>(n) + row_block - 1) / row_block);
    const uint32_t block_dim = static_cast<uint32_t>(std::min<uint64_t>(total_tasks, 65535ull));
    if (block_dim == 0) return ge::GRAPH_FAILED;
    context->SetBlockDim(block_dim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // Workspace: store ksum [B,H,D] float32 so token-parallel blocks can reuse it.
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = static_cast<size_t>(b) * static_cast<size_t>(h) * static_cast<size_t>(d) * sizeof(float);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) return GRAPH_FAILED;
    auto* outShape = context->GetOutputShape(0);
    const auto* qShape = context->GetInputShape(0);
    if (outShape == nullptr || qShape == nullptr) return GRAPH_FAILED;
    *outShape = *qShape; // out: [B,H,N,D]
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
class LinearAttentionCustom : public OpDef {
public:
    explicit LinearAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LinearAttentionCustom);
} // namespace ops
