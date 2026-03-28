
#include "sinkhorn_knopp_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static constexpr uint32_t BLOCK_DIM_MAX = 24;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    SinkhornKnoppCustomTilingData tiling;

    const gert::StorageShape *inShape = context->GetInputShape(0);
    const gert::Shape &origin = inShape->GetOriginShape();
    const uint32_t dimNum = origin.GetDimNum();
    if (dimNum < 2) {
        return ge::GRAPH_FAILED;
    }

    uint32_t B = 1;
    uint32_t N0 = 0, N1 = 0;
    if (dimNum == 2) {
        N0 = static_cast<uint32_t>(origin.GetDim(0));
        N1 = static_cast<uint32_t>(origin.GetDim(1));
    } else {
        B  = static_cast<uint32_t>(origin.GetDim(0));
        N0 = static_cast<uint32_t>(origin.GetDim(dimNum - 2));
        N1 = static_cast<uint32_t>(origin.GetDim(dimNum - 1));
    }
    if (N0 == 0 || N1 == 0 || N0 != N1) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t N = N0;

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *tmaxPtr = attrs->GetAttrPointer<int64_t>(0);
    const float *epsPtr = attrs->GetAttrPointer<float>(1);
    const float *clampPtr = attrs->GetAttrPointer<float>(2);

    const uint32_t totalLength = B * N * N;

    uint32_t blockDim = BLOCK_DIM_MAX;
    if (B > 0 && B < blockDim) blockDim = B;
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    tiling.set_totalLength(totalLength);
    tiling.set_B(B);
    tiling.set_N(N);
    tiling.set_tmax(static_cast<uint32_t>(*tmaxPtr));
    tiling.set_eps(*epsPtr);
    tiling.set_clampMin(*clampPtr);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *in = context->GetInputShape(0);
    gert::Shape *out = context->GetOutputShape(0);
    *out = *in;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType dt = context->GetInputDataType(0);
    context->SetOutputDataType(0, dt);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class SinkhornKnoppCustom : public OpDef {
public:
    explicit SinkhornKnoppCustom(const char *name) : OpDef(name)
    {
        this->Input("logits")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("tmax").AttrType(OPTIONAL).Int(20);
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-8f);
        this->Attr("clamp_min").AttrType(OPTIONAL).Float(0.0f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SinkhornKnoppCustom);
} // namespace ops
