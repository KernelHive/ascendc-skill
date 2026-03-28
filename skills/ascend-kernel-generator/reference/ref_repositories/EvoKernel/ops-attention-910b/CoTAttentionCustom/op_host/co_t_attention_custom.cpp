
#include "co_t_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {
static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CoTAttentionCustomTilingData tiling;

    auto k1_shape  = context->GetInputShape(0)->GetStorageShape();
    auto att_shape = context->GetInputShape(1)->GetStorageShape();
    auto v_shape   = context->GetInputShape(2)->GetStorageShape();

    if (k1_shape.GetDimNum() != 4 || att_shape.GetDimNum() != 3 || v_shape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t bs = static_cast<uint32_t>(k1_shape.GetDim(0));
    const uint32_t C  = static_cast<uint32_t>(k1_shape.GetDim(1));
    const uint32_t H  = static_cast<uint32_t>(k1_shape.GetDim(2));
    const uint32_t W  = static_cast<uint32_t>(k1_shape.GetDim(3));
    if (bs == 0 || C == 0 || H == 0 || W == 0) return ge::GRAPH_FAILED;

    const uint32_t hw = H * W;
    const uint32_t totalRows = bs * C;

    if (static_cast<uint32_t>(att_shape.GetDim(0)) != bs ||
        static_cast<uint32_t>(att_shape.GetDim(1)) != C  ||
        static_cast<uint32_t>(att_shape.GetDim(2)) != hw) {
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint32_t>(v_shape.GetDim(0)) != bs ||
        static_cast<uint32_t>(v_shape.GetDim(1)) != C  ||
        static_cast<uint32_t>(v_shape.GetDim(2)) != hw) {
        return ge::GRAPH_FAILED;
    }

    // Unroll 2 rows per iteration in kernel.
    const uint32_t unroll = 2;

    // Choose a higher core count to reduce per-core serial work and hide MTE gaps.
    // Cap to 48 (common 910B aiv clusters) but safe if runtime provides fewer.
    uint32_t idealCores = 48U;
    uint32_t coreNum = std::min<uint32_t>(idealCores, std::max<uint32_t>(1U, totalRows / 8U));
    coreNum = std::min<uint32_t>(coreNum, totalRows); // cannot exceed rows
    if (coreNum == 0) coreNum = 1;
    context->SetBlockDim(coreNum);

    // Each core gets a contiguous block of rows; make it a multiple of unroll when possible.
    uint32_t blockRows = CeilDivU32(totalRows, coreNum);
    if (blockRows >= unroll) {
        blockRows = CeilDivU32(blockRows, unroll) * unroll;
    }

    tiling.set_bs(bs);
    tiling.set_C(C);
    tiling.set_H(H);
    tiling.set_W(W);
    tiling.set_hw(hw);
    tiling.set_totalRows(totalRows);
    tiling.set_blockRows(blockRows);
    tiling.set_unroll(unroll);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class CoTAttentionCustom : public OpDef {
public:
    explicit CoTAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("k1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("att")
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

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(CoTAttentionCustom);
} // namespace ops
