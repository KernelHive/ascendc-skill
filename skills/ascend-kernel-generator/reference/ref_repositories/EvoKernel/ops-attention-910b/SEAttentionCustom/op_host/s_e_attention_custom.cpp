
#include "se_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SEAttentionCustomTilingData tiling;

    const auto& xShape  = context->GetInputShape(0)->GetStorageShape();
    const auto& w1Shape = context->GetInputShape(1)->GetStorageShape();
    const auto& w2Shape = context->GetInputShape(2)->GetStorageShape();

    if (xShape.GetDimNum() != 4 || w1Shape.GetDimNum() != 2 || w2Shape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    uint32_t B = static_cast<uint32_t>(xShape.GetDim(0));
    uint32_t C = static_cast<uint32_t>(xShape.GetDim(1));
    uint32_t H = static_cast<uint32_t>(xShape.GetDim(2));
    uint32_t W = static_cast<uint32_t>(xShape.GetDim(3));
    if (B == 0 || C == 0 || H == 0 || W == 0) return ge::GRAPH_FAILED;

    // w1: [R, C], w2: [C, R]
    uint32_t R = static_cast<uint32_t>(w1Shape.GetDim(0));
    if (R == 0) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(w1Shape.GetDim(1)) != C) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(w2Shape.GetDim(0)) != C) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(w2Shape.GetDim(1)) != R) return ge::GRAPH_FAILED;

    uint32_t HW = H * W;
    if (HW == 0) return ge::GRAPH_FAILED;

    tiling.set_B(B);
    tiling.set_C(C);
    tiling.set_H(H);
    tiling.set_W(W);
    tiling.set_R(R);
    tiling.set_HW(HW);
    tiling.set_invHW(1.0f / static_cast<float>(HW));

    // Parallelize over batch for correctness and simplicity
    uint32_t block_dim = std::min<uint32_t>(B, 32);
    if (block_dim == 0) block_dim = 1;
    context->SetBlockDim(block_dim);

    // UB-friendly tiling over channels
    // Use multiples of 8 floats (32B). Conservative default 128.
    uint32_t cTile = 128;
    if (cTile > C) {
        cTile = ((C + 7) / 8) * 8;
        if (cTile == 0) cTile = 8;
    }
    tiling.set_cTile(cTile);

    // Align R for UB buffers (vector alignment)
    uint32_t rAlign = ((R + 7) / 8) * 8;
    if (rAlign == 0) rAlign = 8;
    tiling.set_rAlign(rAlign);

    // Temp bytes for Sigmoid implementation; keep safe fixed size.
    tiling.set_sigTmpBytes(16 * 1024);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
class SEAttentionCustom : public OpDef {
public:
    explicit SEAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w2")
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

OP_ADD(SEAttentionCustom);
} // namespace ops
