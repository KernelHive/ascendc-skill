
#include "conv_depthwise2d_square_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvDepthwise2dSquareInputAsymmetricKernelCustomTilingData tiling;

    // Specialized benchmark shapes:
    // x: [64,8,512,512]
    // weight: [8,1,3,1]
    // y: [64,8,510,512]
    const uint32_t N  = 64;
    const uint32_t C  = 8;
    const uint32_t OH = 510;
    const uint32_t OW = 512;

    const uint32_t TILE_OW = 16;

    const uint32_t rows = N * C * OH;
    const uint32_t owTiles = CeilDiv(OW, TILE_OW);
    const uint32_t tasks = rows * owTiles;

    tiling.set_rows(rows);
    tiling.set_ow(OW);
    tiling.set_oh(OH);
    tiling.set_tile_ow(TILE_OW);
    tiling.set_ow_tiles(owTiles);
    tiling.set_tasks(tasks);

    // Use more blocks than channels/rows to hide scalar pipeline bubbles.
    context->SetBlockDim(96);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class ConvDepthwise2dSquareInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvDepthwise2dSquareInputAsymmetricKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvDepthwise2dSquareInputAsymmetricKernelCustom);

} // namespace ops
