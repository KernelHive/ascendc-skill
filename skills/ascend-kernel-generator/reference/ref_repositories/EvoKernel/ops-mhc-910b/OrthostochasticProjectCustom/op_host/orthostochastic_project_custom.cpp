
#include "orthostochastic_project_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static constexpr uint32_t BLOCK_DIM = 1;
static constexpr uint32_t MAX_N = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OrthostochasticProjectCustomTilingData tiling;

    const auto inShape = context->GetInputShape(0)->GetOriginShape();
    if (inShape.GetDimNum() != 2) return ge::GRAPH_FAILED;

    int64_t n0_ = inShape.GetDim(0);
    int64_t n1_ = inShape.GetDim(1);
    if (n0_ <= 0 || n1_ <= 0) return ge::GRAPH_FAILED;
    if (n0_ > MAX_N || n1_ > MAX_N) return ge::GRAPH_FAILED;

    const uint32_t n0 = static_cast<uint32_t>(n0_);
    const uint32_t n1 = static_cast<uint32_t>(n1_);

    const uint32_t transpose = (n0 > n1) ? 1u : 0u;
    const uint32_t m = transpose ? n1 : n0;
    const uint32_t n = transpose ? n0 : n1;

    context->SetBlockDim(BLOCK_DIM);

    tiling.set_n0(n0);
    tiling.set_n1(n1);
    tiling.set_m(m);
    tiling.set_n(n);
    tiling.set_mn(m * n);
    tiling.set_transpose(transpose);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* in = context->GetInputShape(0);
    if (in == nullptr || in->GetDimNum() != 2) return GRAPH_FAILED;
    gert::Shape* out = context->GetOutputShape(0);
    *out = *in;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class OrthostochasticProjectCustom : public OpDef {
public:
    explicit OrthostochasticProjectCustom(const char* name) : OpDef(name)
    {
        this->Input("logits").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("steps"). ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("eps").   ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("a").     ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b").     ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("c").     ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(OrthostochasticProjectCustom);

} // namespace ops
