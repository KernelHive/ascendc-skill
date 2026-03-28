
#include "conv_standard2d_square_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvStandard2dSquareInputSquareKernelCustomTilingData tiling;

    // Specialized fixed shapes:
    // x: [16,16,1024,1024], w: [128,16,3,3], y: [16,128,1022,1022]
    const uint32_t N = 16;
    const uint32_t CIN = 16;
    const uint32_t COUT = 128;
    const uint32_t H = 1024;
    const uint32_t W = 1024;
    const uint32_t K = 3;
    const uint32_t HO = H - K + 1; // 1022
    const uint32_t OW = W - K + 1; // 1022

    const uint32_t TILE_OW = 16;

    const uint32_t rows = N * COUT * HO;
    const uint32_t owTiles = CeilDiv(OW, TILE_OW);
    const uint32_t tasks = rows * owTiles;

    tiling.set_rows(rows);
    tiling.set_ow(OW);
    tiling.set_h(H);
    tiling.set_w(W);
    tiling.set_cin(CIN);
    tiling.set_cout(COUT);
    tiling.set_ho(HO);
    tiling.set_tile_ow(TILE_OW);
    tiling.set_ow_tiles(owTiles);
    tiling.set_tasks(tasks);

    // Moderate fixed parallelism; avoids single-core timeout.
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

class ConvStandard2dSquareInputSquareKernelCustom : public OpDef {
public:
    explicit ConvStandard2dSquareInputSquareKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvStandard2dSquareInputSquareKernelCustom);

} // namespace ops
