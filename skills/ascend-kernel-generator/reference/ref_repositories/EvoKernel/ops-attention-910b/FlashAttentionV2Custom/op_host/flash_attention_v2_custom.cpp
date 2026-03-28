
#include "flash_attention_v2_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    FlashAttentionV2CustomTilingData tiling;

    auto q_shape = context->GetInputShape(0)->GetStorageShape();
    auto k_shape = context->GetInputShape(1)->GetStorageShape();
    auto v_shape = context->GetInputShape(2)->GetStorageShape();
    auto s_shape = context->GetInputShape(3)->GetStorageShape();

    if (q_shape.GetDimNum() != 4 || k_shape.GetDimNum() != 4 || v_shape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t B = static_cast<uint32_t>(q_shape.GetDim(0));
    const uint32_t H = static_cast<uint32_t>(q_shape.GetDim(1));
    const uint32_t S = static_cast<uint32_t>(q_shape.GetDim(2));
    const uint32_t D = static_cast<uint32_t>(q_shape.GetDim(3));
    if (B == 0 || H == 0 || S == 0 || D == 0) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(k_shape.GetDim(0)) != B ||
        static_cast<uint32_t>(k_shape.GetDim(1)) != H ||
        static_cast<uint32_t>(k_shape.GetDim(2)) != S ||
        static_cast<uint32_t>(k_shape.GetDim(3)) != D) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(v_shape.GetDim(0)) != B ||
        static_cast<uint32_t>(v_shape.GetDim(1)) != H ||
        static_cast<uint32_t>(v_shape.GetDim(2)) != S ||
        static_cast<uint32_t>(v_shape.GetDim(3)) != D) return ge::GRAPH_FAILED;

    if (s_shape.GetDimNum() != 1 || static_cast<uint32_t>(s_shape.GetDim(0)) != 1u) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t totalBh = B * H;

    // Keep Ti=1 to reduce UB pressure and allow larger Tj; improves overall memory behavior.
    uint32_t Ti = 1;

    // KV tile size: prefer 64; clamp.
    uint32_t Tj = (S >= 64 ? 64u : (S >= 32 ? 32u : (S >= 16 ? 16u : S)));
    if (Tj == 0) Tj = 1;

    // Parallelize across (B*H). Keep safe cap.
    uint32_t coreNum = std::min<uint32_t>(std::max<uint32_t>(1u, totalBh), 48u);
    context->SetBlockDim(coreNum);

    tiling.set_B(B);
    tiling.set_H(H);
    tiling.set_S(S);
    tiling.set_D(D);
    tiling.set_totalBh(totalBh);
    tiling.set_coreNum(coreNum);
    tiling.set_Ti(Ti);
    tiling.set_Tj(Tj);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
class FlashAttentionV2Custom : public OpDef {
public:
    explicit FlashAttentionV2Custom(const char* name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("o")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(FlashAttentionV2Custom);
} // namespace ops
