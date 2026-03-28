
#include "bam_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static constexpr uint32_t VEC_ALIGN_BYTES = 256;
static constexpr uint32_t FLOAT_BYTES = 4;
static constexpr uint32_t ALIGN_ELEMS = VEC_ALIGN_BYTES / FLOAT_BYTES; // 64

static inline uint32_t AlignDown(uint32_t x, uint32_t a) { return (x / a) * a; }
static inline uint32_t AlignUp(uint32_t x, uint32_t a) { return ((x + a - 1) / a) * a; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    BAMCustomTilingData tiling;

    const auto& xShape = context->GetInputShape(0)->GetStorageShape();
    const auto& cShape = context->GetInputShape(1)->GetStorageShape();
    const auto& sShape = context->GetInputShape(2)->GetStorageShape();

    if (xShape.GetDimNum() == 0 || xShape.GetDimNum() != cShape.GetDimNum() ||
        xShape.GetDimNum() != sShape.GetDimNum()) {
        return ge::GRAPH_FAILED;
    }

    uint32_t dimNum = xShape.GetDimNum();
    uint64_t total64 = 1;
    for (uint32_t i = 0; i < dimNum; ++i) {
        int64_t xd = xShape.GetDim(i);
        int64_t cd = cShape.GetDim(i);
        int64_t sd = sShape.GetDim(i);
        if (xd <= 0 || cd <= 0 || sd <= 0) return ge::GRAPH_FAILED;
        if (xd != cd || xd != sd) return ge::GRAPH_FAILED;
        total64 *= static_cast<uint64_t>(xd);
        if (total64 > 0xFFFFFFFFu) return ge::GRAPH_FAILED;
    }

    uint32_t totalElems = static_cast<uint32_t>(total64);
    if (totalElems == 0) return ge::GRAPH_FAILED;

    // Favor more cores for scalar-bound elementwise kernels; keep a safe cap.
    uint32_t blockDim = std::min<uint32_t>(48, totalElems);
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    uint32_t elemsPerCore = (totalElems + blockDim - 1) / blockDim;
    if (elemsPerCore == 0) elemsPerCore = 1;

    // Larger UB tile to amortize per-tile overhead; single buffering means UB use is modest.
    // 8192 floats = 32KB per queue buffer.
    uint32_t tileElems = 8192;
    tileElems = AlignDown(tileElems, ALIGN_ELEMS);
    if (tileElems < ALIGN_ELEMS) tileElems = ALIGN_ELEMS;

    uint32_t perCoreAlignedDown = AlignDown(elemsPerCore, ALIGN_ELEMS);
    if (perCoreAlignedDown < ALIGN_ELEMS) perCoreAlignedDown = ALIGN_ELEMS;

    if (tileElems > perCoreAlignedDown) tileElems = perCoreAlignedDown;
    if (tileElems < ALIGN_ELEMS) tileElems = ALIGN_ELEMS;

    tiling.set_totalElems(totalElems);
    tiling.set_blockDim(blockDim);
    tiling.set_elemsPerCore(elemsPerCore);
    tiling.set_tileElems(tileElems);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class BAMCustom : public OpDef {
public:
    explicit BAMCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("channel_map")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("spatial_map")
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

OP_ADD(BAMCustom);

} // namespace ops
