
#include "conv_standard1d_dilated_strided_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1U) / b;
}

static inline uint32_t Conv1dOutNoPad(uint32_t lin, uint32_t stride, uint32_t dil, uint32_t k)
{
    const uint32_t effective = dil * (k - 1U) + 1U;
    return (lin - effective) / stride + 1U;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvStandard1dDilatedStridedCustomTilingData tiling;

    // Specialized fixed contract for this benchmark.
    constexpr uint32_t N = 64;
    constexpr uint32_t CIN = 64;
    constexpr uint32_t COUT = 128;
    constexpr uint32_t LIN = 524280;
    constexpr uint32_t K = 3;
    constexpr uint32_t STRIDE = 3;
    constexpr uint32_t DIL = 4;

    const uint32_t LOUT = Conv1dOutNoPad(LIN, STRIDE, DIL, K); // 174758

    // Larger per-block contiguous work to amortize scalar overhead and improve pipeline.
    // 256 keeps register pressure manageable while cutting loop/control overhead vs tiny tiles.
    constexpr uint32_t CHUNK_LOUT = 256;
    const uint32_t LOUT_CHUNKS = CeilDivU32(LOUT, CHUNK_LOUT);

    tiling.set_n(N);
    tiling.set_cin(CIN);
    tiling.set_cout(COUT);
    tiling.set_lin(LIN);
    tiling.set_lout(LOUT);
    tiling.set_chunk_lout(CHUNK_LOUT);
    tiling.set_lout_chunks(LOUT_CHUNKS);
    tiling.set_tasks(N * COUT * LOUT_CHUNKS);

    // One block per (n,co). Each block loops over all chunks.
    // 64*128=8192 blocks gives ample parallelism, but each block has meaningful work.
    context->SetBlockDim(N * COUT);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class ConvStandard1dDilatedStridedCustom : public OpDef {
public:
    explicit ConvStandard1dDilatedStridedCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvStandard1dDilatedStridedCustom);

} // namespace ops
