
#include "tanh_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t x, uint32_t y) { return (x + y - 1u) / y; }

// Tile candidates in elements (fp32), chosen to be reasonably large to reduce per-tile overhead.
static constexpr uint32_t kTileCandidates[] = {32768, 16384, 8192, 4096};

// Keep good occupancy but avoid excessive blocks; runtime will clamp if needed.
static constexpr uint32_t kDefaultBlockDim = 48;

// Conservative UB budget heuristic (leave headroom).
static constexpr uint32_t kUbBudgetBytes = 192 * 1024;

// Temp for tanh tends to be proportional to tile; use a conservative per-element factor plus base.
static constexpr uint32_t kTmpBaseBytes = 1024;
static constexpr uint32_t kTmpPerElemBytes = 2; // conservative; toolchain may use less

// Align temp size to 32B.
static inline uint32_t AlignUp32(uint32_t b) { return (b + 31u) & ~31u; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TanhCustomTilingData tiling;

    const uint32_t totalLength =
        static_cast<uint32_t>(context->GetInputShape(0)->GetOriginShape().GetShapeSize());

    uint32_t blockDim = kDefaultBlockDim;
    if (totalLength == 0) blockDim = 1;
    // Ensure each core has a reasonable amount of work for this bandwidth-bound op.
    const uint32_t minElemsPerBlock = 4096;
    blockDim = std::max(1u, std::min(blockDim, CeilDivU32(std::max(1u, totalLength), minElemsPerBlock)));
    context->SetBlockDim(blockDim);

    // Choose a tile so that each core has at least ~2 tiles when possible (reduces overhead).
    const uint32_t perCore = CeilDivU32(std::max(1u, totalLength), blockDim);
    uint32_t tileSize = kTileCandidates[2]; // default 8192
    if (perCore >= kTileCandidates[0] * 2) tileSize = kTileCandidates[0];
    else if (perCore >= kTileCandidates[1] * 2) tileSize = kTileCandidates[1];
    else if (perCore >= kTileCandidates[2] * 2) tileSize = kTileCandidates[2];
    else tileSize = kTileCandidates[3];

    if (tileSize == 0) tileSize = 1;
    if (tileSize > totalLength && totalLength != 0) tileSize = totalLength;

    // UB model: double-buffered x/y queues + tmp.
    // x+y double buffered => 2*(x+y) = 4*tile*4B = 16*tile bytes.
    // tmp estimated.
    uint32_t tmpSize = AlignUp32(kTmpBaseBytes + kTmpPerElemBytes * tileSize);
    uint32_t need = 16u * tileSize + tmpSize;
    if (need > kUbBudgetBytes) {
        // Reduce tile until within budget.
        for (uint32_t i = 0; i < sizeof(kTileCandidates)/sizeof(kTileCandidates[0]); ++i) {
            uint32_t cand = kTileCandidates[i];
            if (cand > tileSize) continue;
            uint32_t ttmp = AlignUp32(kTmpBaseBytes + kTmpPerElemBytes * cand);
            uint32_t nneed = 16u * cand + ttmp;
            if (nneed <= kUbBudgetBytes) { tileSize = cand; tmpSize = ttmp; break; }
        }
        // Final fallback.
        need = 16u * tileSize + tmpSize;
        if (need > kUbBudgetBytes) {
            tileSize = 1024;
            tmpSize = AlignUp32(kTmpBaseBytes + kTmpPerElemBytes * tileSize);
        }
    }

    tiling.set_totalLength(totalLength);
    tiling.set_blockDim(blockDim);
    tiling.set_tileSize(tileSize);
    tiling.set_tmpSizeBytes(tmpSize);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto dt = context->GetInputDataType(0);
    context->SetOutputDataType(0, dt);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class TanhCustom : public OpDef {
public:
    explicit TanhCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
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

OP_ADD(TanhCustom);

}  // namespace ops
