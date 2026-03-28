
#include "longformer_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    // Expect Q,K,V: [B,H,S,D] float32 ND
    auto qShape = context->GetInputShape(0)->GetStorageShape();
    auto kShape = context->GetInputShape(1)->GetStorageShape();
    auto vShape = context->GetInputShape(2)->GetStorageShape();

    if (qShape.GetDimNum() != 4 || kShape.GetDimNum() != 4 || vShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t b = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t h = static_cast<uint32_t>(qShape.GetDim(1));
    const uint32_t s = static_cast<uint32_t>(qShape.GetDim(2));
    const uint32_t d = static_cast<uint32_t>(qShape.GetDim(3));
    if (b == 0 || h == 0 || s == 0 || d == 0) return ge::GRAPH_FAILED;

    // shape checks: require exact match for Q,K,V
    for (int i = 0; i < 4; ++i) {
        if (static_cast<uint32_t>(kShape.GetDim(i)) != static_cast<uint32_t>(qShape.GetDim(i))) return ge::GRAPH_FAILED;
        if (static_cast<uint32_t>(vShape.GetDim(i)) != static_cast<uint32_t>(qShape.GetDim(i))) return ge::GRAPH_FAILED;
    }

    // Attribute window_size
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) return ge::GRAPH_FAILED;
    const int64_t* wPtr = attrs->GetAttrPointer<int64_t>(0);
    if (wPtr == nullptr) return ge::GRAPH_FAILED;
    int64_t w = *wPtr;
    if (w <= 0) return ge::GRAPH_FAILED;
    if (w > static_cast<int64_t>(s)) w = static_cast<int64_t>(s);

    // Reference model uses fixed globals [0, 511]. Enforce seq_len == 512 for semantic parity.
    if (s != 512U) return ge::GRAPH_FAILED;

    const uint32_t halfW = static_cast<uint32_t>(w / 2);
    const uint32_t maxWin = 2U * halfW + 1U;

    // Row-parallel over queries: pick a small contiguous chunk per core to increase core occupancy.
    // Conservative default to avoid too many blocks: 8 rows/core works well for S=512.
    uint32_t rowsPerCore = 8U;
    if (rowsPerCore > s) rowsPerCore = s;
    const uint32_t totalRows = s;

    LongformerAttentionCustomTilingData tiling;
    tiling.set_b(b);
    tiling.set_h(h);
    tiling.set_s(s);
    tiling.set_d(d);
    tiling.set_window_size(static_cast<uint32_t>(w));
    tiling.set_max_win(maxWin);
    tiling.set_g0(0U);
    tiling.set_g1(511U);
    tiling.set_scale(1.0f / std::sqrt(static_cast<float>(d)));
    tiling.set_total_rows(totalRows);
    tiling.set_rows_per_core(rowsPerCore);

    // blocks = B*H*ceil(S/rowsPerCore)
    const uint32_t chunks = (totalRows + rowsPerCore - 1U) / rowsPerCore;
    const uint32_t block_dim = b * h * chunks;
    if (block_dim == 0) return ge::GRAPH_FAILED;
    context->SetBlockDim(block_dim);

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
    *outShape = *qShape; // output [B,H,S,D]
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class LongformerAttentionCustom : public OpDef {
public:
    explicit LongformerAttentionCustom(const char* name) : OpDef(name)
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

        this->Attr("window_size").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LongformerAttentionCustom);
} // namespace ops
