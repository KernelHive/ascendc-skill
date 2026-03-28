
#include "strided_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    auto qShape = context->GetInputShape(0)->GetStorageShape();
    auto kShape = context->GetInputShape(1)->GetStorageShape();
    auto vShape = context->GetInputShape(2)->GetStorageShape();

    if (qShape.GetDimNum() != 4 || kShape.GetDimNum() != 4 || vShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t b = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t h = static_cast<uint32_t>(qShape.GetDim(1));
    const uint32_t s = static_cast<uint32_t>(qShape.GetDim(2));
    const uint32_t d = static_cast<uint32_t>(qShape.GetDim(3));
    if (b == 0 || h == 0 || s == 0 || d == 0) return ge::GRAPH_FAILED;

    for (int i = 0; i < 4; ++i) {
        if (static_cast<uint32_t>(kShape.GetDim(i)) != static_cast<uint32_t>(qShape.GetDim(i)) ||
            static_cast<uint32_t>(vShape.GetDim(i)) != static_cast<uint32_t>(qShape.GetDim(i))) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) return ge::GRAPH_FAILED;
    const int64_t* stridePtr = attrs->GetAttrPointer<int64_t>(0);
    if (stridePtr == nullptr) return ge::GRAPH_FAILED;
    const int64_t strideAttr = *stridePtr;
    if (strideAttr <= 0) return ge::GRAPH_FAILED;

    const uint32_t stride = static_cast<uint32_t>(strideAttr);
    if (stride == 0 || stride > s) return ge::GRAPH_FAILED;

    // Kernel bounds to keep UB bounded
    if (s > 512) return ge::GRAPH_FAILED;
    if (d > 64)  return ge::GRAPH_FAILED;

    const uint32_t nsel = (s + stride - 1) / stride;
    if (nsel == 0 || nsel > 512) return ge::GRAPH_FAILED;

    const uint64_t totalRows64 = static_cast<uint64_t>(b) * h * s;
    if (totalRows64 == 0 || totalRows64 > 0xFFFFFFFFULL) return ge::GRAPH_FAILED;
    const uint32_t totalRows = static_cast<uint32_t>(totalRows64);

    // Choose a modest core count to increase parallelism over rows.
    // 910B has many cores; keep conservative to avoid oversubscription.
    uint32_t coreNum = std::min<uint32_t>(totalRows, 96U);
    if (coreNum == 0) coreNum = 1;

    const uint32_t rowsPerCore = (totalRows + coreNum - 1) / coreNum;

    StridedAttentionCustomTilingData tiling;
    tiling.set_b(b);
    tiling.set_h(h);
    tiling.set_s(s);
    tiling.set_d(d);
    tiling.set_stride(stride);
    tiling.set_nsel(nsel);
    tiling.set_scale(1.0f / std::sqrt(static_cast<float>(d)));
    tiling.set_totalRows(totalRows);
    tiling.set_rowsPerCore(rowsPerCore);

    context->SetBlockDim(coreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    auto* outShape = context->GetOutputShape(0);
    const auto* qShape = context->GetInputShape(0);
    if (outShape == nullptr || qShape == nullptr) return GRAPH_FAILED;
    *outShape = *qShape; // output [B,H,S,D]
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class StridedAttentionCustom : public OpDef {
public:
    explicit StridedAttentionCustom(const char* name) : OpDef(name)
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

        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("stride").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(StridedAttentionCustom);
} // namespace ops
