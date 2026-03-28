
#include "coord_att_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <algorithm>

namespace optiling {
static constexpr uint32_t BLOCK_BYTES = 32;   // 32B alignment for float vector ops
static constexpr uint32_t TYPE_BYTES  = 4;    // float32
static constexpr uint32_t ALIGN_ELEMS = BLOCK_BYTES / TYPE_BYTES; // 8
static constexpr uint32_t MAX_CORE_NUM_910B = 32;

// UB bytes (BUFFER_NUM=1): x + aw + ah + tmp + y = 5 * tileElems * 4B = 20*tileElems.
// tileElems=4096 => 80KB, safe on 910B.
static constexpr uint32_t DEFAULT_TILE_ELEMS = 4096;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CoordAttCustomTilingData tiling;

    auto x_shape  = context->GetInputShape(0)->GetStorageShape();
    auto aw_shape = context->GetInputShape(1)->GetStorageShape();
    auto ah_shape = context->GetInputShape(2)->GetStorageShape();

    if (x_shape.GetDimNum() == 0) return ge::GRAPH_FAILED;
    if (x_shape.GetDimNum() != aw_shape.GetDimNum() || x_shape.GetDimNum() != ah_shape.GetDimNum()) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < static_cast<uint32_t>(x_shape.GetDimNum()); ++i) {
        if (x_shape.GetDim(i) != aw_shape.GetDim(i) || x_shape.GetDim(i) != ah_shape.GetDim(i)) {
            return ge::GRAPH_FAILED;
        }
    }

    const uint64_t total64 = x_shape.GetShapeSize();
    if (total64 == 0 || total64 > UINT32_MAX) return ge::GRAPH_FAILED;
    const uint32_t totalElems = static_cast<uint32_t>(total64);

    // Elementwise: use enough cores but avoid too tiny per-core work.
    // Target ~64KB of x per core => 16K floats/core.
    uint32_t blockDim = (totalElems + 16384 - 1) / 16384;
    blockDim = std::max<uint32_t>(1, blockDim);
    blockDim = std::min<uint32_t>(MAX_CORE_NUM_910B, blockDim);

    const uint32_t maxUseful = std::max<uint32_t>(1, (totalElems + ALIGN_ELEMS - 1) / ALIGN_ELEMS);
    if (blockDim > maxUseful) blockDim = maxUseful;

    context->SetBlockDim(blockDim);

    uint32_t coreNominal = (totalElems + blockDim - 1) / blockDim;
    uint32_t coreElemsAligned = (coreNominal / ALIGN_ELEMS) * ALIGN_ELEMS;
    if (coreElemsAligned == 0) coreElemsAligned = ALIGN_ELEMS;

    uint32_t tileElems = DEFAULT_TILE_ELEMS;
    tileElems = (tileElems / ALIGN_ELEMS) * ALIGN_ELEMS;
    if (tileElems == 0) tileElems = ALIGN_ELEMS;
    if (tileElems > coreElemsAligned) tileElems = coreElemsAligned;

    tiling.set_totalElems(totalElems);
    tiling.set_blockDim(blockDim);
    tiling.set_coreElemsAligned(coreElemsAligned);
    tiling.set_tileElems(tileElems);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class CoordAttCustom : public OpDef {
public:
    explicit CoordAttCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("a_w")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("a_h")
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

OP_ADD(CoordAttCustom);
} // namespace ops
