
#include "cbam_block_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CBAMBlockCustomTilingData tiling;

    const auto& xShape  = context->GetInputShape(0)->GetStorageShape();
    const auto& caShape = context->GetInputShape(1)->GetStorageShape();
    const auto& saShape = context->GetInputShape(2)->GetStorageShape();

    if (xShape.GetDimNum() != 4 || caShape.GetDimNum() != 4 || saShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    int64_t b64 = xShape.GetDim(0);
    int64_t c64 = xShape.GetDim(1);
    int64_t h64 = xShape.GetDim(2);
    int64_t w64 = xShape.GetDim(3);
    if (b64 <= 0 || c64 <= 0 || h64 <= 0 || w64 <= 0) return ge::GRAPH_FAILED;

    uint32_t B = static_cast<uint32_t>(b64);
    uint32_t C = static_cast<uint32_t>(c64);
    uint32_t H = static_cast<uint32_t>(h64);
    uint32_t W = static_cast<uint32_t>(w64);
    uint32_t HW = H * W;
    if (HW == 0) return ge::GRAPH_FAILED;

    // ca: [B, C, 1, 1]
    if (caShape.GetDim(0) != b64 || caShape.GetDim(1) != c64 ||
        caShape.GetDim(2) != 1  || caShape.GetDim(3) != 1) {
        return ge::GRAPH_FAILED;
    }
    // sa: [B, 1, H, W]
    if (saShape.GetDim(0) != b64 || saShape.GetDim(1) != 1 ||
        saShape.GetDim(2) != h64 || saShape.GetDim(3) != w64) {
        return ge::GRAPH_FAILED;
    }

    uint64_t strideB64 = static_cast<uint64_t>(C) * static_cast<uint64_t>(HW);
    if (strideB64 == 0 || strideB64 > 0xFFFFFFFFu) return ge::GRAPH_FAILED;
    uint32_t strideB = static_cast<uint32_t>(strideB64);

    tiling.set_B(B);
    tiling.set_C(C);
    tiling.set_H(H);
    tiling.set_W(W);
    tiling.set_HW(HW);
    tiling.set_strideB(strideB);

    // Batch parallelism across cores.
    uint32_t block_dim = std::min<uint32_t>(B, 32);
    if (block_dim == 0) block_dim = 1;
    context->SetBlockDim(block_dim);

    // Channel tile (pure scalar GM access in kernel; tiling helps locality).
    uint32_t cTile = 128;
    if (cTile > C) {
        cTile = ((C + 7) / 8) * 8;
        if (cTile == 0) cTile = 8;
    }
    tiling.set_cTile(cTile);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
class CBAMBlockCustom : public OpDef {
public:
    explicit CBAMBlockCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("ca")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("sa")
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

OP_ADD(CBAMBlockCustom);
} // namespace ops
