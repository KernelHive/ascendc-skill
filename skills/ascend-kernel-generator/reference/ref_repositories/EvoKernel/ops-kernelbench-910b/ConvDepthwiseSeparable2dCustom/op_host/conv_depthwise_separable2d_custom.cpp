
#include "conv_depthwise_separable2d_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1U) / b;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvDepthwiseSeparable2dCustomTilingData tiling;

    // Specialized contract fixed for this benchmark.
    const uint32_t N = 16;
    const uint32_t CIN = 64;
    const uint32_t COUT = 128;
    const uint32_t H = 512;
    const uint32_t W = 512;

    // Depthwise 3x3, stride=1, pad=1, dilation=1 => OH=OW=512.
    const uint32_t OH = 512;
    const uint32_t OW = 512;

    const uint32_t TILE_OW = 8;
    const uint32_t OW_TILES = CeilDivU32(OW, TILE_OW);

    const uint32_t CO_TILE = 16; // compute 16 output channels per task (improves pw weight reuse)
    const uint32_t CO_TILES = CeilDivU32(COUT, CO_TILE);

    // For pad=1, kw=3: need 0 <= ow-1 and ow-1+(TILE_OW-1)+2 < W => ow in [1, W-(TILE_OW+1)]
    const uint32_t owInteriorStart = 1U;
    const uint32_t owInteriorEnd = (W > (TILE_OW + 1U)) ? (W - (TILE_OW + 1U) + 1U) : 0U; // exclusive

    // For pad=1, kh=3: need 0 <= oh-1 and oh+1 < H => oh in [1, H-1)
    const uint32_t ohInteriorStart = 1U;
    const uint32_t ohInteriorEnd = (H >= 2U) ? (H - 1U) : 0U; // exclusive

    tiling.set_tasks(N * OH * OW_TILES * CO_TILES);

    tiling.set_n(N);
    tiling.set_cin(CIN);
    tiling.set_cout(COUT);
    tiling.set_h(H);
    tiling.set_w(W);
    tiling.set_oh(OH);
    tiling.set_ow(OW);

    tiling.set_tile_ow(TILE_OW);
    tiling.set_ow_tiles(OW_TILES);

    tiling.set_co_tile(CO_TILE);
    tiling.set_co_tiles(CO_TILES);

    tiling.set_ow_interior_start(owInteriorStart);
    tiling.set_ow_interior_end(owInteriorEnd);
    tiling.set_oh_interior_start(ohInteriorStart);
    tiling.set_oh_interior_end(ohInteriorEnd);

    // Keep stable moderate parallelism; each task now does more work (16 co).
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

class ConvDepthwiseSeparable2dCustom : public OpDef {
public:
    explicit ConvDepthwiseSeparable2dCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("w_depthwise")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("w_pointwise")
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

OP_ADD(ConvDepthwiseSeparable2dCustom);

} // namespace ops
