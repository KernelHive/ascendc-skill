
#include "conv_standard2d_square_input_asymmetric_kernel_dilated_padded_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1U) / b;
}

static inline uint32_t ConvOut(uint32_t in, uint32_t pad, uint32_t dil, uint32_t k, uint32_t stride)
{
    return (in + 2U * pad - dil * (k - 1U) - 1U) / stride + 1U;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvStandard2dSquareInputAsymmetricKernelDilatedPaddedCustomTilingData tiling;

    // Fixed specialization contract:
    const uint32_t N = 8;
    const uint32_t CIN = 32;
    const uint32_t COUT = 64;
    const uint32_t H = 512;
    const uint32_t W = 512;
    const uint32_t KH = 5;
    const uint32_t KW = 9;

    const uint32_t STRIDE_H = 1;
    const uint32_t STRIDE_W = 1;
    const uint32_t PAD_H = 2;
    const uint32_t PAD_W = 4;
    const uint32_t DIL_H = 2;
    const uint32_t DIL_W = 3;

    const uint32_t OH = ConvOut(H, PAD_H, DIL_H, KH, STRIDE_H); // 508
    const uint32_t OW = ConvOut(W, PAD_W, DIL_W, KW, STRIDE_W); // 496

    const uint32_t TILE_OW = 8;
    const uint32_t OW_TILES = CeilDivU32(OW, TILE_OW);

    const uint32_t COBLOCK = 2;
    const uint32_t CO_BLOCKS = COUT / COBLOCK; // 32

    // Interior region in OW coordinates for lane l in [0,7]:
    // iw0 = ow - PAD_W + l
    // taps at iw0 + s*DIL_W, s=0..8, require:
    //   0 <= iw0  and  iw0 + 8*DIL_W < W
    // Strong for all lanes in tile: l=0 and l=7
    //   ow >= PAD_W
    //   ow - PAD_W + 7 + 8*DIL_W < W  => ow < W + PAD_W - 7 - 8*DIL_W
    // with W=512, PAD_W=4, DIL_W=3 => ow < 485
    const uint32_t OW_INTERIOR_BEGIN = PAD_W; // 4
    const uint32_t OW_INTERIOR_END = (W + PAD_W > (7 + 8 * DIL_W)) ? (W + PAD_W - (7 + 8 * DIL_W)) : 0; // 485

    // Convert interior OW coordinate range to full tiles:
    // Need whole tile [ow0, ow0+TILE_OW) inside [OW_INTERIOR_BEGIN, OW_INTERIOR_END)
    const uint32_t interiorTileBegin = CeilDivU32(OW_INTERIOR_BEGIN, TILE_OW);
    const uint32_t interiorTileEnd = (OW_INTERIOR_END >= TILE_OW) ? (OW_INTERIOR_END / TILE_OW) : 0;

    tiling.set_ow(OW);
    tiling.set_oh(OH);
    tiling.set_w(W);
    tiling.set_h(H);
    tiling.set_cin(CIN);
    tiling.set_cout(COUT);
    tiling.set_tile_ow(TILE_OW);
    tiling.set_ow_tiles(OW_TILES);
    tiling.set_coblock(COBLOCK);
    tiling.set_co_blocks(CO_BLOCKS);
    tiling.set_interior_tile_begin(interiorTileBegin);
    tiling.set_interior_tile_end(interiorTileEnd);

    tiling.set_tasks(N * OH * CO_BLOCKS * OW_TILES);

    // Keep stable scheduling.
    context->SetBlockDim(64);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class ConvStandard2dSquareInputAsymmetricKernelDilatedPaddedCustom : public OpDef {
public:
    explicit ConvStandard2dSquareInputAsymmetricKernelDilatedPaddedCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvStandard2dSquareInputAsymmetricKernelDilatedPaddedCustom);

} // namespace ops
