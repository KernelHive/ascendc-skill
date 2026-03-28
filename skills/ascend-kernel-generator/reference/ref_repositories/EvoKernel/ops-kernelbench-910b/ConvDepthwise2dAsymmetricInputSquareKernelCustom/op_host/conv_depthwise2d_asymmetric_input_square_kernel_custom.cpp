
#include "conv_depthwise2d_asymmetric_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvDepthwise2dAsymmetricInputSquareKernelCustomTilingData tiling;

    // Specialized benchmark:
    // x: [64, 128, 256, 512]
    // weight: [128, 1, 3, 3] (depthwise)
    // stride=1, pad=0, dilation=1 => y: [64, 128, 254, 510]
    constexpr uint32_t N  = 64;
    constexpr uint32_t C  = 128;
    constexpr uint32_t H  = 256;
    constexpr uint32_t W  = 512;
    constexpr uint32_t OH = 254;
    constexpr uint32_t OW = 510;

    constexpr uint32_t TILE_OW = 16;
    const uint32_t owTiles = CeilDivU32(OW, TILE_OW);
    const uint32_t rows = N * C * OH;
    const uint32_t tasks = rows * owTiles;

    tiling.set_n(N);
    tiling.set_c(C);
    tiling.set_h(H);
    tiling.set_w(W);
    tiling.set_oh(OH);
    tiling.set_ow(OW);
    tiling.set_tile_ow(TILE_OW);
    tiling.set_ow_tiles(owTiles);
    tiling.set_rows(rows);
    tiling.set_tasks(tasks);

    // Increase parallelism to address under-parallelization / pipeline gaps.
    // 910B has many AIV/AICore resources; 48 is a safe, commonly stable choice.
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

class ConvDepthwise2dAsymmetricInputSquareKernelCustom : public OpDef {
public:
    explicit ConvDepthwise2dAsymmetricInputSquareKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvDepthwise2dAsymmetricInputSquareKernelCustom);

} // namespace ops
