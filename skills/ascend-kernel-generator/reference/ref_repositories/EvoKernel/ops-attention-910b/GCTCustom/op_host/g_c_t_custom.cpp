
#include "gct_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static inline uint32_t AlignUp(uint32_t v, uint32_t a) {
    return (v + a - 1U) / a * a;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GCTCustomTilingData tiling;

    const auto& xShape   = context->GetInputShape(0)->GetStorageShape();
    const auto& cShape   = context->GetInputShape(1)->GetStorageShape();
    const auto& eShape   = context->GetInputShape(2)->GetStorageShape();

    if (xShape.GetDimNum() != 4) return ge::GRAPH_FAILED;
    // c and eps must be scalar / [1]
    if (cShape.GetShapeSize() != 1) return ge::GRAPH_FAILED;
    if (eShape.GetShapeSize() != 1) return ge::GRAPH_FAILED;

    uint32_t N = static_cast<uint32_t>(xShape.GetDim(0));
    uint32_t C = static_cast<uint32_t>(xShape.GetDim(1));
    uint32_t H = static_cast<uint32_t>(xShape.GetDim(2));
    uint32_t W = static_cast<uint32_t>(xShape.GetDim(3));
    if (N == 0 || C == 0 || H == 0 || W == 0) return ge::GRAPH_FAILED;

    uint32_t HW = H * W;
    uint32_t CHW = C * HW;
    uint32_t totalLength = N * CHW;

    // vector alignment: 32B => 8 floats
    uint32_t CAlign = AlignUp(C, 8U);
    if (CAlign == 0) CAlign = 8U;

    tiling.set_N(N);
    tiling.set_C(C);
    tiling.set_H(H);
    tiling.set_W(W);
    tiling.set_HW(HW);
    tiling.set_CHW(CHW);
    tiling.set_totalLength(totalLength);
    tiling.set_CAlign(CAlign);
    tiling.set_invC(1.0f / static_cast<float>(C));

    // Parallelize over batch (cap to avoid too many tiny cores)
    uint32_t block_dim = std::min<uint32_t>(N, 32U);
    if (block_dim == 0) block_dim = 1U;
    context->SetBlockDim(block_dim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
class GCTCustom : public OpDef {
public:
    explicit GCTCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("eps")
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

OP_ADD(GCTCustom);
} // namespace ops
