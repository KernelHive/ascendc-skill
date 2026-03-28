
#include "conv_standard3d_asymmetric_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvStandard3dAsymmetricInputAsymmetricKernelCustomTilingData tiling;

    // Specialized fixed shapes for this kernel.
    constexpr uint32_t N = 8;
    constexpr uint32_t COUT = 64;
    constexpr uint32_t DOUT = 14;
    constexpr uint32_t HOUT = 124;
    constexpr uint32_t totalRows = N * COUT * DOUT * HOUT;

    // Increase parallelism moderately to hide scalar latency/pipeline gaps.
    uint32_t blockDim = 128;
    blockDim = std::max(1u, blockDim);
    context->SetBlockDim(blockDim);

    tiling.set_blockDim(blockDim);
    tiling.set_totalRows(totalRows);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class ConvStandard3dAsymmetricInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvStandard3dAsymmetricInputAsymmetricKernelCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("weight")
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

OP_ADD(ConvStandard3dAsymmetricInputAsymmetricKernelCustom);

} // namespace ops
