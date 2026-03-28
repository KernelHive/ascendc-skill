
#include "stream_weighted_sum_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static inline uint32_t AlignDown(uint32_t x, uint32_t a) { return (a == 0) ? x : (x / a) * a; }

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    StreamWeightedSumCustomTilingData tiling;

    const gert::StorageShape *xS = context->GetInputShape(0); // (B,T,N,C)
    const gert::StorageShape *wS = context->GetInputShape(1); // (B,T,N)
    if (xS == nullptr || wS == nullptr) return ge::GRAPH_FAILED;

    const gert::Shape &x = xS->GetOriginShape();
    const gert::Shape &w = wS->GetOriginShape();
    if (x.GetDimNum() != 4 || w.GetDimNum() != 3) return ge::GRAPH_FAILED;

    const uint32_t B = static_cast<uint32_t>(x.GetDim(0));
    const uint32_t T = static_cast<uint32_t>(x.GetDim(1));
    const uint32_t N = static_cast<uint32_t>(x.GetDim(2));
    const uint32_t C = static_cast<uint32_t>(x.GetDim(3));

    if (B == 0 || T == 0 || N == 0 || C == 0) return ge::GRAPH_FAILED;

    if (static_cast<uint32_t>(w.GetDim(0)) != B ||
        static_cast<uint32_t>(w.GetDim(1)) != T ||
        static_cast<uint32_t>(w.GetDim(2)) != N) {
        return ge::GRAPH_FAILED;
    }

    auto *in0 = context->GetInputTensor(0);
    auto *in1 = context->GetInputTensor(1);
    if (in0 == nullptr || in1 == nullptr) return ge::GRAPH_FAILED;
    if (in0->GetDataType() != ge::DT_FLOAT || in1->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t BT = B * T;

    // Prefer full C when reasonable; ensure multiple-of-8 and divisor of C to avoid tail in hot path.
    uint32_t cTile = 512;
    if (C < cTile) cTile = AlignDown(C, 8);
    if (cTile == 0) cTile = 8;

    if (C % cTile != 0) {
        const uint32_t cand[] = {512, 256, 128, 64, 32, 16, 8};
        for (uint32_t i = 0; i < sizeof(cand)/sizeof(cand[0]); ++i) {
            if (C >= cand[i] && (C % cand[i] == 0)) { cTile = cand[i]; break; }
        }
        if (C % cTile != 0) cTile = AlignDown(C, 8);
        if (cTile == 0) cTile = 8;
    }

    uint32_t tilesPerRow = (cTile == 0) ? 0 : (C / cTile);
    uint32_t numTiles = BT * tilesPerRow;

    // Launch cores based on tile count (better device fill when BT is small but C is large).
    uint32_t maxCores = 48;
    uint32_t blockDim = (numTiles < maxCores) ? numTiles : maxCores;
    if (blockDim == 0) blockDim = 1;
    if (blockDim > 65535) blockDim = 65535;
    context->SetBlockDim(blockDim);

    tiling.set_B(B);
    tiling.set_T(T);
    tiling.set_N(N);
    tiling.set_C(C);
    tiling.set_BT(BT);
    tiling.set_cTile(cTile);
    tiling.set_tilesPerRow(tilesPerRow);
    tiling.set_numTiles(numTiles);

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
    const gert::Shape *x = context->GetInputShape(0); // (B,T,N,C)
    const gert::Shape *w = context->GetInputShape(1); // (B,T,N)
    gert::Shape *out = context->GetOutputShape(0);    // (B,T,C)
    if (x == nullptr || w == nullptr || out == nullptr) return GRAPH_FAILED;
    if (x->GetDimNum() != 4 || w->GetDimNum() != 3) return GRAPH_FAILED;

    out->SetDimNum(3);
    out->SetDim(0, x->GetDim(0));
    out->SetDim(1, x->GetDim(1));
    out->SetDim(2, x->GetDim(3));
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

class StreamWeightedSumCustom : public OpDef {
public:
    explicit StreamWeightedSumCustom(const char *name) : OpDef(name)
    {
        this->Input("x_stream")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("weights")
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

OP_ADD(StreamWeightedSumCustom);

} // namespace ops
