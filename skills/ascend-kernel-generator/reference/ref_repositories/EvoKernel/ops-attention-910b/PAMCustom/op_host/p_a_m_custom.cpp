
#include "pam_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static constexpr uint32_t VEC_ALIGN_BYTES = 256;
static constexpr uint32_t FLOAT_BYTES = 4;
static constexpr uint32_t ALIGN_ELEMS = VEC_ALIGN_BYTES / FLOAT_BYTES; // 64

static inline uint32_t AlignUp(uint32_t x, uint32_t a) { return ((x + a - 1) / a) * a; }
static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    PAMCustomTilingData tiling;

    const auto& bShape = context->GetInputShape(0)->GetStorageShape(); // [N,S,C]
    const auto& cShape = context->GetInputShape(1)->GetStorageShape(); // [N,C,S]
    const auto& dShape = context->GetInputShape(2)->GetStorageShape(); // [N,S,C]

    if (bShape.GetDimNum() != 3 || cShape.GetDimNum() != 3 || dShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    int64_t n64 = bShape.GetDim(0);
    int64_t s64 = bShape.GetDim(1);
    int64_t c64 = bShape.GetDim(2);
    if (n64 <= 0 || s64 <= 0 || c64 <= 0) return ge::GRAPH_FAILED;

    if (cShape.GetDim(0) != n64 || cShape.GetDim(1) != c64 || cShape.GetDim(2) != s64) return ge::GRAPH_FAILED;
    if (dShape.GetDim(0) != n64 || dShape.GetDim(1) != s64 || dShape.GetDim(2) != c64) return ge::GRAPH_FAILED;

    if (s64 > 64) return ge::GRAPH_FAILED;
    if (c64 > 1024) return ge::GRAPH_FAILED;
    if ((static_cast<uint32_t>(c64) % 16u) != 0u) return ge::GRAPH_FAILED;

    uint32_t N = static_cast<uint32_t>(n64);
    uint32_t S = static_cast<uint32_t>(s64);
    uint32_t C = static_cast<uint32_t>(c64);

    uint32_t S_pad = AlignUp(S, ALIGN_ELEMS);
    if (S_pad != 64) return ge::GRAPH_FAILED;

    const uint32_t totalRows = N * S;

    // More parallelism than mapping-by-batch: map across (N*S) query rows.
    // Clamp conservatively; 910B commonly tolerates 24-48 well for small kernels.
    uint32_t blockDim = std::min<uint32_t>(48u, std::max<uint32_t>(1u, totalRows));
    context->SetBlockDim(blockDim);

    const uint32_t rowsPerCore = CeilDivU32(totalRows, blockDim);

    // Channel tile to trade UB and fewer GM touches; keep small and aligned.
    uint32_t cTile = 128;
    if (C < 128) cTile = 64;
    if (C < 64)  cTile = 32;
    if (C < 32)  cTile = 16;
    if (cTile > C) cTile = C;
    // keep 16-aligned for vector-friendly DataCopy sizes (C is 16-aligned, but tail tile may be smaller)
    if (cTile >= 16) cTile = (cTile / 16u) * 16u;
    if (cTile == 0) cTile = C;

    tiling.set_N(N);
    tiling.set_S(S);
    tiling.set_C(C);
    tiling.set_S_pad(S_pad);
    tiling.set_totalRows(totalRows);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_cTile(cTile);
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

class PAMCustom : public OpDef {
public:
    explicit PAMCustom(const char* name) : OpDef(name)
    {
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("d")
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

OP_ADD(PAMCustom);

} // namespace ops
