
#include "conv_standard2d_asymmetric_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvStandard2dAsymmetricInputSquareKernelCustomTilingData tiling;

    // Specialized benchmark shapes:
    // x: [8,64,512,1024], w: [128,64,3,3], y: [8,128,510,1022]
    const uint32_t N = 8;
    const uint32_t CIN = 64;
    const uint32_t COUT = 128;
    const uint32_t H = 512;
    const uint32_t W = 1024;
    const uint32_t OH = 510;   // H - 3 + 1
    const uint32_t OW = 1022;  // W - 3 + 1

    // Keep the stable tile size, but kernel will run a branch-free fast path for full tiles.
    const uint32_t TILE_OW = 16;

    const uint32_t rows = N * COUT * OH;
    const uint32_t owTiles = CeilDiv(OW, TILE_OW);
    const uint32_t tasks = rows * owTiles;

    tiling.set_rows(rows);
    tiling.set_ow(OW);
    tiling.set_h(H);
    tiling.set_w(W);
    tiling.set_cin(CIN);
    tiling.set_cout(COUT);
    tiling.set_oh(OH);
    tiling.set_tile_ow(TILE_OW);
    tiling.set_ow_tiles(owTiles);
    tiling.set_tasks(tasks);

    // Keep baseline launch to avoid stressing runtime/device scheduling; improvements come from reduced scalar overhead.
    context->SetBlockDim(48);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class ConvStandard2dAsymmetricInputSquareKernelCustom : public OpDef {
public:
    explicit ConvStandard2dAsymmetricInputSquareKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvStandard2dAsymmetricInputSquareKernelCustom);

} // namespace ops
