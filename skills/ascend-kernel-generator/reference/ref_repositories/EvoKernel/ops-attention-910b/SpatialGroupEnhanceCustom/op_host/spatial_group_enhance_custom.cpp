
#include "spatial_group_enhance_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>
#include <cstdint>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SpatialGroupEnhanceCustomTilingData tiling;

    const auto& xShape = context->GetInputShape(0)->GetStorageShape();
    const auto& wShape = context->GetInputShape(1)->GetStorageShape();
    const auto& bShape = context->GetInputShape(2)->GetStorageShape();

    if (xShape.GetDimNum() != 4) return ge::GRAPH_FAILED;
    if (wShape.GetDimNum() != 4) return ge::GRAPH_FAILED;
    if (bShape.GetDimNum() != 4) return ge::GRAPH_FAILED;

    uint32_t B = static_cast<uint32_t>(xShape.GetDim(0));
    uint32_t C = static_cast<uint32_t>(xShape.GetDim(1));
    uint32_t H = static_cast<uint32_t>(xShape.GetDim(2));
    uint32_t W = static_cast<uint32_t>(xShape.GetDim(3));
    if (B == 0 || C == 0 || H == 0 || W == 0) return ge::GRAPH_FAILED;

    // Require weight/bias shapes exactly [1, G, 1, 1]
    if (static_cast<uint32_t>(wShape.GetDim(0)) != 1u ||
        static_cast<uint32_t>(wShape.GetDim(2)) != 1u ||
        static_cast<uint32_t>(wShape.GetDim(3)) != 1u) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(bShape.GetDim(0)) != 1u ||
        static_cast<uint32_t>(bShape.GetDim(2)) != 1u ||
        static_cast<uint32_t>(bShape.GetDim(3)) != 1u) return ge::GRAPH_FAILED;

    uint32_t G = static_cast<uint32_t>(wShape.GetDim(1));
    if (G == 0) return ge::GRAPH_FAILED;
    if (static_cast<uint32_t>(bShape.GetDim(1)) != G) return ge::GRAPH_FAILED;
    if (C % G != 0) return ge::GRAPH_FAILED;

    uint32_t Cg = C / G;
    if (Cg == 0) return ge::GRAPH_FAILED;

    uint32_t HW = H * W;
    if (HW == 0) return ge::GRAPH_FAILED;

    uint32_t groupsTotal = B * G;
    if (groupsTotal == 0) return ge::GRAPH_FAILED;

    tiling.set_B(B);
    tiling.set_C(C);
    tiling.set_H(H);
    tiling.set_W(W);
    tiling.set_G(G);
    tiling.set_Cg(Cg);
    tiling.set_HW(HW);
    tiling.set_groupsTotal(groupsTotal);

    uint32_t hwAlign = ((HW + 7) / 8) * 8;
    if (hwAlign == 0) hwAlign = 8;
    tiling.set_hwAlign(hwAlign);

    tiling.set_invHW(1.0f / static_cast<float>(HW));
    tiling.set_invCg(1.0f / static_cast<float>(Cg));
    tiling.set_epsilon(1e-5f);

    // Conservative fixed size temp buffer for Sigmoid
    tiling.set_sigTmpBytes(16 * 1024);

    // Parallelize over (B*G), cap for portability
    uint32_t block_dim = std::min<uint32_t>(groupsTotal, 65535u);
    if (block_dim == 0) block_dim = 1;
    context->SetBlockDim(block_dim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {
class SpatialGroupEnhanceCustom : public OpDef {
public:
    explicit SpatialGroupEnhanceCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SpatialGroupEnhanceCustom);
} // namespace ops
