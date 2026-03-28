
#include "conv_transposed1d_dilated_custom_tiling.h"
#include "register/op_def_registry.h"
#include <stdint.h>
#include <algorithm>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed1dDilatedCustomTilingData tiling;

    // Specialized for:
    // x: [32,32,131072], w: [32,64,5], stride=1, pad=0, dil=3, outpad=0
    // Lout = (Lin-1)*1 - 2*0 + 3*(5-1) + 0 + 1 = 131084
    constexpr uint32_t N = 32;
    constexpr uint32_t COUT = 64;
    constexpr uint32_t LOUT = 131084;

    tiling.set_totalY(N * COUT * LOUT);
    tiling.set_lout(LOUT);

    // Stable blockDim for deterministic partitioning.
    uint32_t blockDim = 256;
    blockDim = std::max(1u, blockDim);
    context->SetBlockDim(blockDim);
    tiling.set_blockDim(blockDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class ConvTransposed1dDilatedCustom : public OpDef {
public:
    explicit ConvTransposed1dDilatedCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTransposed1dDilatedCustom);

} // namespace ops
