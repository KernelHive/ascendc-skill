
#include "optimized_flash_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>
#include <cmath>

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

    const uint32_t b  = static_cast<uint32_t>(qShape.GetDim(0));
    const uint32_t h  = static_cast<uint32_t>(qShape.GetDim(1));
    const uint32_t sq = static_cast<uint32_t>(qShape.GetDim(2));
    const uint32_t d  = static_cast<uint32_t>(qShape.GetDim(3));

    const uint32_t bk = static_cast<uint32_t>(kShape.GetDim(0));
    const uint32_t hk = static_cast<uint32_t>(kShape.GetDim(1));
    const uint32_t sk = static_cast<uint32_t>(kShape.GetDim(2));
    const uint32_t dk = static_cast<uint32_t>(kShape.GetDim(3));

    if (b == 0 || h == 0 || sq == 0 || sk == 0 || d == 0) return ge::GRAPH_FAILED;
    if (bk != b || hk != h || dk != d) return ge::GRAPH_FAILED;

    const uint32_t bv = static_cast<uint32_t>(vShape.GetDim(0));
    const uint32_t hv = static_cast<uint32_t>(vShape.GetDim(1));
    const uint32_t sv = static_cast<uint32_t>(vShape.GetDim(2));
    const uint32_t dv = static_cast<uint32_t>(vShape.GetDim(3));
    if (bv != b || hv != h || sv != sk || dv != d) return ge::GRAPH_FAILED;

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) return ge::GRAPH_FAILED;
    const bool* hasBiasPtr = attrs->GetAttrPointer<bool>(0);
    if (hasBiasPtr == nullptr) return ge::GRAPH_FAILED;
    const bool hasBias = *hasBiasPtr;

    uint32_t biasBroadcastB = 0;
    if (hasBias) {
        auto biasShape = context->GetInputShape(3)->GetStorageShape();
        if (biasShape.GetDimNum() == 3) {
            if (static_cast<uint32_t>(biasShape.GetDim(0)) != h ||
                static_cast<uint32_t>(biasShape.GetDim(1)) != sq ||
                static_cast<uint32_t>(biasShape.GetDim(2)) != sk) {
                return ge::GRAPH_FAILED;
            }
            biasBroadcastB = 1; // [H,Sq,Sk]
        } else if (biasShape.GetDimNum() == 4) {
            if (static_cast<uint32_t>(biasShape.GetDim(0)) != b ||
                static_cast<uint32_t>(biasShape.GetDim(1)) != h ||
                static_cast<uint32_t>(biasShape.GetDim(2)) != sq ||
                static_cast<uint32_t>(biasShape.GetDim(3)) != sk) {
                return ge::GRAPH_FAILED;
            }
            biasBroadcastB = 0; // [B,H,Sq,Sk]
        } else {
            return ge::GRAPH_FAILED;
        }
    }

    OptimizedFlashAttentionCustomTilingData tiling;
    tiling.set_b(b);
    tiling.set_h(h);
    tiling.set_sq(sq);
    tiling.set_sk(sk);
    tiling.set_d(d);
    tiling.set_scale(1.0f / std::sqrt(static_cast<float>(d)));
    tiling.set_hasBias(hasBias ? 1u : 0u);
    tiling.set_biasBroadcastB(biasBroadcastB);

    // Choose a safe tile for D to reduce scalar GM accesses; keep power-of-2 when possible.
    uint32_t tileD = 64;
    if (d < tileD) tileD = d;
    if (tileD == 0) tileD = 1;
    tiling.set_tileD(tileD);

    // Parallelize across query rows.
    const uint64_t block64 = static_cast<uint64_t>(b) * h * sq;
    if (block64 == 0 || block64 > 0x7FFFFFFFul) return ge::GRAPH_FAILED;
    context->SetBlockDim(static_cast<uint32_t>(block64));

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
    *outShape = *qShape; // [B,H,Sq,D]
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {
class OptimizedFlashAttentionCustom : public OpDef {
public:
    explicit OptimizedFlashAttentionCustom(const char* name) : OpDef(name)
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
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("has_bias").Bool();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(OptimizedFlashAttentionCustom);
} // namespace ops
