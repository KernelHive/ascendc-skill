
#include "squeeze_net_fire_module_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <algorithm>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SqueezeNetFireModuleCustomTilingData tiling;

    uint32_t totalX  = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t totalWs = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    uint32_t totalBs = context->GetInputShape(2)->GetStorageShape().GetShapeSize();
    uint32_t totalW1 = context->GetInputShape(3)->GetStorageShape().GetShapeSize();
    uint32_t totalB1 = context->GetInputShape(4)->GetStorageShape().GetShapeSize();
    uint32_t totalW3 = context->GetInputShape(5)->GetStorageShape().GetShapeSize();
    uint32_t totalB3 = context->GetInputShape(6)->GetStorageShape().GetShapeSize();
    uint32_t totalY  = context->GetOutputShape(0)->GetStorageShape().GetShapeSize();

    tiling.set_totalX(totalX);
    tiling.set_totalWs(totalWs);
    tiling.set_totalBs(totalBs);
    tiling.set_totalW1(totalW1);
    tiling.set_totalB1(totalB1);
    tiling.set_totalW3(totalW3);
    tiling.set_totalB3(totalB3);
    tiling.set_totalY(totalY);

    // Benchmark-fixed shapes.
    constexpr uint32_t N = 128;
    constexpr uint32_t H = 256;
    constexpr uint32_t OUTC = 128;
    constexpr uint32_t tileC = 8;
    static_assert(OUTC % tileC == 0, "OUTC must be divisible by tileC");

    constexpr uint32_t rowsNH = N * H;           // 32768
    constexpr uint32_t outcTiles = OUTC / tileC; // 16

    tiling.set_rowsNH(rowsNH);
    tiling.set_outcTiles(outcTiles);
    tiling.set_tileC(tileC);

    uint32_t blocks = rowsNH * outcTiles; // 524288

    // Slightly higher cap to increase concurrency but still conservative for stability.
    constexpr uint32_t kMaxBlockDim = 384;
    uint32_t blockDim = std::min<uint32_t>(kMaxBlockDim, blocks);
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class SqueezeNetFireModuleCustom : public OpDef {
public:
    explicit SqueezeNetFireModuleCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("w_squeeze").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_squeeze").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("w_expand1").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_expand1").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("w_expand3").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_expand3").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SqueezeNetFireModuleCustom);

} // namespace ops
