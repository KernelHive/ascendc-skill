
#include "grouped_gemm_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <algorithm>

namespace optiling {

static constexpr uint32_t MAX_K = 8192;
static constexpr uint32_t MAX_N = 8192;
static constexpr uint32_t MAX_G = 4096;

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    auto lhsShape = context->GetInputShape(0)->GetStorageShape();
    auto rhsShape = context->GetInputShape(1)->GetStorageShape();
    auto idxShape = context->GetInputShape(2)->GetStorageShape();

    if (lhsShape.GetDimNum() != 2 || rhsShape.GetDimNum() != 3 || idxShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t m = static_cast<uint32_t>(lhsShape.GetDim(0));
    const uint32_t k = static_cast<uint32_t>(lhsShape.GetDim(1));
    const uint32_t g = static_cast<uint32_t>(rhsShape.GetDim(0));
    const uint32_t n = static_cast<uint32_t>(rhsShape.GetDim(1));
    const uint32_t k2 = static_cast<uint32_t>(rhsShape.GetDim(2));
    const uint64_t idxN = static_cast<uint64_t>(idxShape.GetDim(0));

    if (m == 0 || k == 0 || n == 0 || g == 0) return ge::GRAPH_FAILED;
    if (k2 != k) return ge::GRAPH_FAILED;
    if (idxN != static_cast<uint64_t>(m)) return ge::GRAPH_FAILED;

    if (k > MAX_K || n > MAX_N || g > MAX_G) return ge::GRAPH_FAILED;

    // Adaptive but safe tiles. Goal: reduce scalar overhead while keeping UB bounded.
    // nTile primarily controls parallelism and RHS packing cost; kTile controls reuse.
    uint32_t nTile = 128;
    if (n < 128) nTile = 64;
    if (n < 64)  nTile = 32;
    if (n < 32)  nTile = n;

    uint32_t kTile = 256;
    if (k < 256) kTile = 128;
    if (k < 128) kTile = 64;
    if (k < 64)  kTile = k;

    GroupedGEMMCustomTilingData td;
    td.set_m(m);
    td.set_k(k);
    td.set_n(n);
    td.set_g(g);
    td.set_nTile(nTile);
    td.set_kTile(kTile);

    const uint32_t tilesPerRow = CeilDivU32(n, nTile);
    td.set_tilesPerRow(tilesPerRow);
    td.set_totalTiles(m * tilesPerRow);

    // Keep enough parallelism; cap for stability across environments.
    uint32_t blockDim = td.get_totalTiles();
    if (blockDim == 0) blockDim = 1;
    blockDim = std::min<uint32_t>(blockDim, 96u);
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
    const auto* lhsShape = context->GetInputShape(0);
    const auto* rhsShape = context->GetInputShape(1);
    if (outShape == nullptr || lhsShape == nullptr || rhsShape == nullptr) return GRAPH_FAILED;

    outShape->SetDimNum(2);
    outShape->SetDim(0, lhsShape->GetDim(0));
    outShape->SetDim(1, rhsShape->GetDim(1));
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
class GroupedGEMMCustom : public OpDef {
public:
    explicit GroupedGEMMCustom(const char* name) : OpDef(name)
    {
        this->Input("lhs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("rhs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("m_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
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

OP_ADD(GroupedGEMMCustom);
} // namespace ops
