
#include "double_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    DoubleAttentionCustomTilingData tiling;

    const auto& aShape = context->GetInputShape(0)->GetStorageShape(); // [B,Cm,HW]
    const auto& bShape = context->GetInputShape(1)->GetStorageShape(); // [B,Cn,HW]
    const auto& vShape = context->GetInputShape(2)->GetStorageShape(); // [B,Cn,HW]

    if (aShape.GetDimNum() != 3 || bShape.GetDimNum() != 3 || vShape.GetDimNum() != 3) return ge::GRAPH_FAILED;

    int64_t B64  = aShape.GetDim(0);
    int64_t Cm64 = aShape.GetDim(1);
    int64_t HW64 = aShape.GetDim(2);

    if (B64 <= 0 || Cm64 <= 0 || HW64 <= 0) return ge::GRAPH_FAILED;

    if (bShape.GetDim(0) != B64 || vShape.GetDim(0) != B64) return ge::GRAPH_FAILED;
    int64_t Cn64 = bShape.GetDim(1);
    if (Cn64 <= 0) return ge::GRAPH_FAILED;
    if (bShape.GetDim(2) != HW64) return ge::GRAPH_FAILED;
    if (vShape.GetDim(1) != Cn64 || vShape.GetDim(2) != HW64) return ge::GRAPH_FAILED;

    // Guard against unexpected storage.
    const uint64_t aElems = aShape.GetShapeSize();
    const uint64_t bElems = bShape.GetShapeSize();
    const uint64_t vElems = vShape.GetShapeSize();
    if (aElems != (uint64_t)B64 * (uint64_t)Cm64 * (uint64_t)HW64) return ge::GRAPH_FAILED;
    if (bElems != (uint64_t)B64 * (uint64_t)Cn64 * (uint64_t)HW64) return ge::GRAPH_FAILED;
    if (vElems != (uint64_t)B64 * (uint64_t)Cn64 * (uint64_t)HW64) return ge::GRAPH_FAILED;

    uint32_t B  = static_cast<uint32_t>(B64);
    uint32_t Cm = static_cast<uint32_t>(Cm64);
    uint32_t Cn = static_cast<uint32_t>(Cn64);
    uint32_t HW = static_cast<uint32_t>(HW64);

    tiling.set_B(B);
    tiling.set_Cm(Cm);
    tiling.set_Cn(Cn);
    tiling.set_HW(HW);

    // Parallelize across batch. Cap block dim for portability here.
    uint32_t block_dim = std::min<uint32_t>(B, 32);
    if (block_dim == 0) block_dim = 1;
    context->SetBlockDim(block_dim);

    uint32_t batchesPerCore = (B + block_dim - 1) / block_dim;
    if (batchesPerCore == 0) batchesPerCore = 1;
    tiling.set_batchesPerCore(batchesPerCore);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
class DoubleAttentionCustom : public OpDef {
public:
    explicit DoubleAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(DoubleAttentionCustom);
} // namespace ops
