
#include "gc_module_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static inline uint32_t AlignUp(uint32_t v, uint32_t a) {
    return (v + a - 1U) / a * a;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GCModuleCustomTilingData tiling;

    const auto& xShape = context->GetInputShape(0)->GetStorageShape();
    const auto& yShape = context->GetInputShape(1)->GetStorageShape();

    if (xShape.GetDimNum() != 4 || yShape.GetDimNum() != 4) return ge::GRAPH_FAILED;

    uint32_t N = static_cast<uint32_t>(xShape.GetDim(0));
    uint32_t C = static_cast<uint32_t>(xShape.GetDim(1));
    uint32_t H = static_cast<uint32_t>(xShape.GetDim(2));
    uint32_t W = static_cast<uint32_t>(xShape.GetDim(3));
    if (N == 0 || C == 0 || H == 0 || W == 0) return ge::GRAPH_FAILED;

    // y must be [N,C,1,1]
    if (static_cast<uint32_t>(yShape.GetDim(0)) != N ||
        static_cast<uint32_t>(yShape.GetDim(1)) != C ||
        static_cast<uint32_t>(yShape.GetDim(2)) != 1U ||
        static_cast<uint32_t>(yShape.GetDim(3)) != 1U) {
        return ge::GRAPH_FAILED;
    }

    uint64_t total64 = xShape.GetShapeSize();
    if (total64 == 0 || total64 > UINT32_MAX) return ge::GRAPH_FAILED;
    uint32_t totalElems = static_cast<uint32_t>(total64);

    uint32_t HW = H * W;
    if (HW == 0) return ge::GRAPH_FAILED;

    // Parallelism: cap at 32 for 910B like the reference.
    uint32_t block_dim = std::min<uint32_t>(32U, std::max<uint32_t>(1U, N));
    context->SetBlockDim(block_dim);

    // 32B alignment: 8 floats
    constexpr uint32_t ALIGN = 8U;
    uint32_t alignedTotal = AlignUp(totalElems, ALIGN);

    uint32_t blockElems = (alignedTotal + block_dim - 1U) / block_dim;
    blockElems = AlignUp(blockElems, ALIGN);
    if (blockElems == 0) blockElems = ALIGN;

    // UB tile: 4096 floats (16KB) aligned; clamp to blockElems.
    uint32_t tileElems = 4096U;
    tileElems = (tileElems / ALIGN) * ALIGN;
    if (tileElems == 0) tileElems = ALIGN;
    if (tileElems > blockElems) tileElems = blockElems;

    tiling.set_N(N);
    tiling.set_C(C);
    tiling.set_H(H);
    tiling.set_W(W);
    tiling.set_HW(HW);
    tiling.set_totalElems(totalElems);
    tiling.set_blockElems(blockElems);
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
class GCModuleCustom : public OpDef {
public:
    explicit GCModuleCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y_bc11")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GCModuleCustom);
} // namespace ops
