
#include "da_module_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    DAModuleCustomTilingData tiling;

    // p_out: [bs, hw, c]
    // c_out: [bs, c, hw]
    auto p_shape = context->GetInputShape(0)->GetStorageShape();
    auto c_shape = context->GetInputShape(1)->GetStorageShape();
    if (p_shape.GetDimNum() != 3 || c_shape.GetDimNum() != 3) return ge::GRAPH_FAILED;

    const uint32_t bs = static_cast<uint32_t>(p_shape.GetDim(0));
    const uint32_t hw = static_cast<uint32_t>(p_shape.GetDim(1));
    const uint32_t c  = static_cast<uint32_t>(p_shape.GetDim(2));

    if (static_cast<uint32_t>(c_shape.GetDim(0)) != bs ||
        static_cast<uint32_t>(c_shape.GetDim(1)) != c  ||
        static_cast<uint32_t>(c_shape.GetDim(2)) != hw) return ge::GRAPH_FAILED;

    // Channel tile: keep UB usage small and stable.
    // UB needs ~ (pRow + cSeg + outSeg) * 4 bytes = (cTile + cTile + cTile)*4 = 12*cTile bytes.
    // cTile=128 => 1.5KB, very safe.
    uint32_t cTile = 128;
    if (c < 128) cTile = 64;
    if (c < 64)  cTile = 32;
    if (c < 32)  cTile = 16;
    if (c < 16)  cTile = 8;
    if (cTile > c) cTile = c;

    const uint32_t totalRows = bs * hw; // rows are (b,pos)

    // Use more cores than baseline to reduce serialization, but avoid oversubscription.
    // 910B commonly supports many blocks; clamp conservatively to 24.
    uint32_t blockDim = std::min(std::max(1u, totalRows), 24u);
    context->SetBlockDim(blockDim);

    const uint32_t rowsPerCore = CeilDivU32(totalRows, blockDim);

    tiling.set_bs(bs);
    tiling.set_c(c);
    tiling.set_hw(hw);
    tiling.set_totalRows(totalRows);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_cTile(cTile);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
class DAModuleCustom : public OpDef {
public:
    explicit DAModuleCustom(const char* name) : OpDef(name)
    {
        this->Input("p_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("c_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("h").Int();
        this->Attr("w").Int();

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(DAModuleCustom);
} // namespace ops
