
#include "conv3d_mish_tanh_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv3dMishTanhCustomTilingData tiling;

    const auto xShape = context->GetInputShape(0)->GetStorageShape();
    if (xShape.GetDimNum() != 5) return ge::GRAPH_FAILED; // expect NCDHW

    const uint64_t total64 = static_cast<uint64_t>(xShape.GetShapeSize());
    if (total64 == 0 || total64 > 0xFFFFFFFFull) return ge::GRAPH_FAILED;
    const uint32_t total = static_cast<uint32_t>(total64);
    tiling.set_total_x(total);

    // Moderate core count; avoid excessive stream pressure
    uint32_t blockDim = 32;
    if (blockDim > total) blockDim = total;
    if (blockDim == 0) blockDim = 1;

    uint32_t elemsPerBlock = (total + blockDim - 1u) / blockDim;
    if (elemsPerBlock == 0) elemsPerBlock = 1;

    tiling.set_block_dim(blockDim);
    tiling.set_elems_per_block(elemsPerBlock);

    context->SetBlockDim(blockDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class Conv3dMishTanhCustom : public OpDef {
public:
    explicit Conv3dMishTanhCustom(const char* name) : OpDef(name)
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

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv3dMishTanhCustom);

} // namespace ops
