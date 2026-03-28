
#include "mhc_update_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstdint>

namespace optiling {

static inline uint32_t RoundDownTo(uint32_t x, uint32_t m) { return (m == 0) ? x : (x / m) * m; }

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    MhcUpdateCustomTilingData tiling;

    const gert::StorageShape *xsS = context->GetInputShape(0); // (B,T,J,C)
    const gert::StorageShape *hpS = context->GetInputShape(1); // (B,T,I)
    const gert::StorageShape *hrS = context->GetInputShape(2); // (B,T,I,J)
    const gert::StorageShape *yS  = context->GetInputShape(3); // (B,T,C)
    if (xsS == nullptr || hpS == nullptr || hrS == nullptr || yS == nullptr) return ge::GRAPH_FAILED;

    const gert::Shape &xs = xsS->GetOriginShape();
    const gert::Shape &hp = hpS->GetOriginShape();
    const gert::Shape &hr = hrS->GetOriginShape();
    const gert::Shape &y  = yS->GetOriginShape();

    if (xs.GetDimNum() != 4 || hp.GetDimNum() != 3 || hr.GetDimNum() != 4 || y.GetDimNum() != 3) return ge::GRAPH_FAILED;

    uint32_t B = (uint32_t)xs.GetDim(0);
    uint32_t T = (uint32_t)xs.GetDim(1);
    uint32_t J = (uint32_t)xs.GetDim(2);
    uint32_t C = (uint32_t)xs.GetDim(3);

    uint32_t Bhp = (uint32_t)hp.GetDim(0);
    uint32_t Thp = (uint32_t)hp.GetDim(1);
    uint32_t I   = (uint32_t)hp.GetDim(2);

    uint32_t Bhr = (uint32_t)hr.GetDim(0);
    uint32_t Thr = (uint32_t)hr.GetDim(1);
    uint32_t Ihr = (uint32_t)hr.GetDim(2);
    uint32_t Jhr = (uint32_t)hr.GetDim(3);

    uint32_t By  = (uint32_t)y.GetDim(0);
    uint32_t Ty  = (uint32_t)y.GetDim(1);
    uint32_t Cy  = (uint32_t)y.GetDim(2);

    if (B == 0 || T == 0 || I == 0 || J == 0 || C == 0) return ge::GRAPH_FAILED;
    if (B != Bhp || T != Thp || B != Bhr || T != Thr || B != By || T != Ty) return ge::GRAPH_FAILED;
    if (I != Ihr || J != Jhr || C != Cy) return ge::GRAPH_FAILED;

    const uint32_t BT = B * T;

    // Prefer a single-tile for common aligned C (e.g., 256) to remove loop & DMA launch overhead.
    uint32_t Vc = 0;
    if ((C % 8) == 0) {
        Vc = C;
    } else {
        // Fallback: choose largest aligned tile.
        if (C >= 256) Vc = 256;
        else if (C >= 128) Vc = 128;
        else if (C >= 64) Vc = 64;
        else if (C >= 32) Vc = 32;
        else if (C >= 16) Vc = 16;
        else Vc = 8;
        Vc = RoundDownTo(Vc, 8);
        if (Vc < 8) Vc = 8;
        if (Vc > C) Vc = RoundDownTo(C, 8);
        if (Vc < 8) Vc = 8;
    }

    // Parallelize over (b,t).
    uint32_t blockDim = 24;
    if (BT < blockDim) blockDim = BT;
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    tiling.set_B(B);
    tiling.set_T(T);
    tiling.set_I(I);
    tiling.set_J(J);
    tiling.set_C(C);
    tiling.set_BT(BT);
    tiling.set_Vc(Vc);

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
    const gert::Shape *xs = context->GetInputShape(0); // (B,T,J,C)
    const gert::Shape *hp = context->GetInputShape(1); // (B,T,I)
    gert::Shape *out = context->GetOutputShape(0);     // (B,T,I,C)
    if (xs == nullptr || hp == nullptr || out == nullptr) return GRAPH_FAILED;
    if (xs->GetDimNum() != 4 || hp->GetDimNum() != 3) return GRAPH_FAILED;

    out->SetDimNum(4);
    out->SetDim(0, xs->GetDim(0));
    out->SetDim(1, xs->GetDim(1));
    out->SetDim(2, hp->GetDim(2));
    out->SetDim(3, xs->GetDim(3));
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

class MhcUpdateCustom : public OpDef {
public:
    explicit MhcUpdateCustom(const char *name) : OpDef(name)
    {
        this->Input("x_stream").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("h_post").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("h_res").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MhcUpdateCustom);

} // namespace ops
