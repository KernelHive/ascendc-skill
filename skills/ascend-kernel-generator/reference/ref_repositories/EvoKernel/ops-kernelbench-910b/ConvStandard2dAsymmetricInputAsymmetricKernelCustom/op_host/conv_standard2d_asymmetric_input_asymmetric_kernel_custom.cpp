
#include "conv_standard2d_asymmetric_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1U) / b;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvStandard2dAsymmetricInputAsymmetricKernelCustomTilingData tiling;

    // Specialization contract:
    // x: [8,64,512,256], w: [128,64,5,7] (OIHW), y: [8,128,508,250]
    const uint32_t N = 8;
    const uint32_t CIN = 64;
    const uint32_t COUT = 128;
    const uint32_t H = 512;
    const uint32_t W = 256;
    const uint32_t KH = 5;
    const uint32_t KW = 7;

    const uint32_t OH = H - KH + 1; // 508
    const uint32_t OW = W - KW + 1; // 250

    // Slightly larger tile to amortize scalar overhead, still reasonable register pressure.
    const uint32_t TILE_OW = 16;
    const uint32_t OW_TILES = CeilDivU32(OW, TILE_OW);

    const uint32_t rows = N * COUT * OH;
    const uint32_t tasks = rows * OW_TILES;

    tiling.set_rows(rows);
    tiling.set_ow(OW);
    tiling.set_h(H);
    tiling.set_w(W);
    tiling.set_cin(CIN);
    tiling.set_cout(COUT);
    tiling.set_oh(OH);
    tiling.set_tile_ow(TILE_OW);
    tiling.set_ow_tiles(OW_TILES);
    tiling.set_tasks(tasks);

    // Increase parallelism safely for 910B: more independent tiles reduce pipeline gaps.
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

class ConvStandard2dAsymmetricInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvStandard2dAsymmetricInputAsymmetricKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvStandard2dAsymmetricInputAsymmetricKernelCustom);

} // namespace ops
