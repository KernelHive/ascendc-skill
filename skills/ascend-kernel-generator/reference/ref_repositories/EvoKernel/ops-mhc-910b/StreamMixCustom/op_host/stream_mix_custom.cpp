
#include "stream_mix_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static inline uint32_t RoundDownTo(uint32_t x, uint32_t a) {
    return (a == 0) ? x : (x / a) * a;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    StreamMixCustomTilingData tiling;

    const gert::StorageShape *xS = context->GetInputShape(0); // (B,T,N,C)
    const gert::StorageShape *hS = context->GetInputShape(1); // (B,T,N,N)
    if (xS == nullptr || hS == nullptr) return ge::GRAPH_FAILED;

    const gert::Shape &x = xS->GetOriginShape();
    const gert::Shape &h = hS->GetOriginShape();
    if (x.GetDimNum() != 4 || h.GetDimNum() != 4) return ge::GRAPH_FAILED;

    uint32_t B = static_cast<uint32_t>(x.GetDim(0));
    uint32_t T = static_cast<uint32_t>(x.GetDim(1));
    uint32_t N = static_cast<uint32_t>(x.GetDim(2));
    uint32_t C = static_cast<uint32_t>(x.GetDim(3));
    if (B == 0 || T == 0 || N == 0 || C == 0) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(h.GetDim(0)) != B ||
        static_cast<uint32_t>(h.GetDim(1)) != T ||
        static_cast<uint32_t>(h.GetDim(2)) != N ||
        static_cast<uint32_t>(h.GetDim(3)) != N) return ge::GRAPH_FAILED;

    auto *in0 = context->GetInputTensor(0);
    auto *in1 = context->GetInputTensor(1);
    if (in0 == nullptr || in1 == nullptr) return ge::GRAPH_FAILED;
    if (in0->GetDataType() != ge::DT_FLOAT || in1->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    uint32_t BT = B * T;

    // Parallelize over (b,t) rows; keep weight reuse and low scalar overhead.
    uint32_t maxCores = 40;
    uint32_t blockDim = (BT < maxCores) ? BT : maxCores;
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    // Choose cTile for N==4 path. Favor aligned, long vectors (multiple of 128).
    // Cap for UB safety; for C=256 -> 256.
    uint32_t cTile = C;
    if (cTile > 512) cTile = 512;
    cTile = RoundDownTo(cTile, 128);
    if (cTile == 0) cTile = (C >= 128) ? 128 : C;
    if (cTile > C) cTile = C;
    if (cTile == 0) cTile = 1;

    tiling.set_B(B);
    tiling.set_T(T);
    tiling.set_N(N);
    tiling.set_C(C);
    tiling.set_BT(BT);
    tiling.set_cTile(cTile);

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
    const gert::Shape *x = context->GetInputShape(0);
    gert::Shape *out = context->GetOutputShape(0);
    if (x == nullptr || out == nullptr) return GRAPH_FAILED;
    if (x->GetDimNum() != 4) return GRAPH_FAILED;
    *out = *x;
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

class StreamMixCustom : public OpDef {
public:
    explicit StreamMixCustom(const char *name) : OpDef(name)
    {
        this->Input("x_stream")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("h_res")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(StreamMixCustom);

} // namespace ops
