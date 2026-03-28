
#include "cross_entropy_loss_custom_tiling.h"
#include "register/op_def_registry.h"
#include <stdint.h>

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (b == 0U) ? 0U : (a + b - 1U) / b;
}

static inline uint32_t AlignDownU32(uint32_t v, uint32_t a) {
    return (a == 0U) ? v : (v / a) * a;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CrossEntropyLossCustomTilingData tiling;

    uint64_t predElems64 = static_cast<uint64_t>(context->GetInputShape(0)->GetStorageShape().GetShapeSize());
    uint64_t tgtElems64  = static_cast<uint64_t>(context->GetInputShape(1)->GetStorageShape().GetShapeSize());

    uint32_t N = static_cast<uint32_t>(tgtElems64);
    uint32_t C = 0U;
    if (N > 0U) {
        C = static_cast<uint32_t>(predElems64 / static_cast<uint64_t>(N));
    }

    // Keep single-core for robustness (no cross-core sync/reduction).
    context->SetBlockDim(1);

    // Increase tileC to reduce tilesPerRow and DataCopy/Reduce invocations.
    // Keep multiple of 16 for better alignment.
    uint32_t tileC = 2048U;
    if (C > 0U && tileC > C) tileC = C;
    tileC = AlignDownU32(tileC, 16U);
    if (tileC == 0U) tileC = (C >= 16U) ? 16U : (C == 0U ? 1U : C);

    uint32_t tilesPerRow = CeilDivU32(C, tileC);

    tiling.set_N(N);
    tiling.set_C(C);
    tiling.set_tileC(tileC);
    tiling.set_tilesPerRow(tilesPerRow);
    tiling.set_invN((N > 0U) ? (1.0f / static_cast<float>(N)) : 0.0f);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
class CrossEntropyLossCustom : public OpDef {
public:
    explicit CrossEntropyLossCustom(const char* name) : OpDef(name)
    {
        this->Input("predict")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("target")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
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

OP_ADD(CrossEntropyLossCustom);
} // namespace ops
