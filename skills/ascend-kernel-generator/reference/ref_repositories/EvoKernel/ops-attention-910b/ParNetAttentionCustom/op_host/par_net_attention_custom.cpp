
#include "par_net_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

// Alignment friendly for vector instructions: 256B.
// float32 => 64 elems per 256B.
static constexpr uint32_t VEC_ALIGN_BYTES = 256;
static constexpr uint32_t FLOAT_BYTES = 4;
static constexpr uint32_t ALIGN_ELEMS = VEC_ALIGN_BYTES / FLOAT_BYTES; // 64

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ParNetAttentionCustomTilingData tiling;

    const auto& aShape = context->GetInputShape(0)->GetStorageShape();
    const auto& bShape = context->GetInputShape(1)->GetStorageShape();
    const auto& cShape = context->GetInputShape(2)->GetStorageShape();

    // Require same rank and same shape for all inputs.
    if (aShape.GetDimNum() == 0 || aShape.GetDimNum() != bShape.GetDimNum() ||
        aShape.GetDimNum() != cShape.GetDimNum()) {
        return ge::GRAPH_FAILED;
    }

    uint32_t dimNum = aShape.GetDimNum();
    uint64_t total64 = 1;
    for (uint32_t i = 0; i < dimNum; ++i) {
        int64_t ad = aShape.GetDim(i);
        int64_t bd = bShape.GetDim(i);
        int64_t cd = cShape.GetDim(i);
        if (ad <= 0 || bd <= 0 || cd <= 0) return ge::GRAPH_FAILED;
        if (ad != bd || ad != cd) return ge::GRAPH_FAILED;
        total64 *= static_cast<uint64_t>(ad);
        if (total64 > 0xFFFFFFFFu) return ge::GRAPH_FAILED; // keep 32-bit indexing in kernel
    }

    uint32_t totalElems = static_cast<uint32_t>(total64);
    if (totalElems == 0) return ge::GRAPH_FAILED;

    // Parallelize across cores (cap to 32 for conservative portability).
    uint32_t blockDim = std::min<uint32_t>(32, totalElems);
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    uint32_t elemsPerCore = (totalElems + blockDim - 1) / blockDim;
    if (elemsPerCore == 0) elemsPerCore = 1;

    // UB tile: choose multiple of ALIGN_ELEMS.
    // Keep conservative: 4096 floats (~16KB) per tensor buffer.
    // We need x1,x2,x3,sum,sigmoid,y and tmp => keep tile modest.
    uint32_t tileElems = 4096;
    tileElems = (tileElems / ALIGN_ELEMS) * ALIGN_ELEMS;
    if (tileElems < ALIGN_ELEMS) tileElems = ALIGN_ELEMS;

    if (tileElems > elemsPerCore) {
        tileElems = ((elemsPerCore + ALIGN_ELEMS - 1) / ALIGN_ELEMS) * ALIGN_ELEMS;
        if (tileElems == 0) tileElems = ALIGN_ELEMS;
    }

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

class ParNetAttentionCustom : public OpDef {
public:
    explicit ParNetAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("x3")
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

OP_ADD(ParNetAttentionCustom);

} // namespace ops
