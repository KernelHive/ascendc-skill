
#include "conv_transposed2d_asymmetric_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"
#include <stdint.h>
#include <algorithm>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed2dAsymmetricInputAsymmetricKernelCustomTilingData tiling;

    // Specialized for:
    // x: [64,64,128,256] fp32 NCHW
    // w: [64,128,3,5] fp32 [Cin,Cout,Kh,Kw]
    // y: [64,128,130,260]
    constexpr uint32_t N = 64;
    constexpr uint32_t COUT = 128;
    constexpr uint32_t HOUT = 130;
    constexpr uint32_t WOUT = 260;
    constexpr uint32_t TILE_WO = 8;

    constexpr uint32_t tilesPerRow = (WOUT + TILE_WO - 1) / TILE_WO; // 33
    constexpr uint32_t totalRows = N * COUT * HOUT;
    constexpr uint32_t totalTiles = totalRows * tilesPerRow;

    tiling.set_totalTiles(totalTiles);
    tiling.set_tilesPerRow(tilesPerRow);
    tiling.set_wout(WOUT);
    tiling.set_tileWo(TILE_WO);

    // Increase parallelism; keep stable/predictable.
    uint32_t blockDim = 512;
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

class ConvTransposed2dAsymmetricInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvTransposed2dAsymmetricInputAsymmetricKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTransposed2dAsymmetricInputAsymmetricKernelCustom);

} // namespace ops
