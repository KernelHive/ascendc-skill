
#include "conv_standard2d_square_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvStandard2dSquareInputAsymmetricKernelCustomTilingData tiling;

    // Specialized fixed shapes:
    // x: [8,32,512,512], w: [64,32,5,9], y: [8,64,508,504]
    const uint32_t N = 8;
    const uint32_t CIN = 32;
    const uint32_t COUT = 64;
    const uint32_t H = 512;
    const uint32_t W = 512;
    const uint32_t KH = 5;
    const uint32_t KW = 9;
    const uint32_t HO = H - KH + 1; // 508
    const uint32_t WO = W - KW + 1; // 504

    const uint32_t TILE_WO = 16;

    const uint32_t rows = N * COUT * HO;
    const uint32_t woTiles = CeilDiv(WO, TILE_WO);
    const uint32_t tasks = rows * woTiles;

    tiling.set_rows(rows);
    tiling.set_wo(WO);
    tiling.set_h(H);
    tiling.set_w(W);
    tiling.set_cin(CIN);
    tiling.set_cout(COUT);
    tiling.set_ho(HO);
    tiling.set_tile_wo(TILE_WO);
    tiling.set_wo_tiles(woTiles);
    tiling.set_tasks(tasks);

    // More parallelism than row-wise mapping, but keep launch moderate/stable.
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

class ConvStandard2dSquareInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvStandard2dSquareInputAsymmetricKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvStandard2dSquareInputAsymmetricKernelCustom);

} // namespace ops
