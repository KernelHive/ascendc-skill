
#include "matmul_with_small_k_dimension_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto shape_a = context->GetInputTensor(0)->GetOriginShape(); // A: (M,K)
    auto shape_b = context->GetInputTensor(1)->GetOriginShape(); // B: (K,N)

    if (shape_a.GetDimNum() != 2 || shape_b.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int32_t M  = static_cast<int32_t>(shape_a.GetDim(0));
    const int32_t K1 = static_cast<int32_t>(shape_a.GetDim(1));
    const int32_t K2 = static_cast<int32_t>(shape_b.GetDim(0));
    const int32_t N  = static_cast<int32_t>(shape_b.GetDim(1));
    if (K1 != K2) {
        return ge::GRAPH_FAILED;
    }
    const int32_t K = K1;

    // Small-K specialization hint:
    // Keep K tile small and focus on larger M/N tiles for better reuse.
    // Avoid invalid/unspecified K split (do NOT pass -1).
    const int32_t baseM = 128;
    const int32_t baseN = 128;
    const int32_t baseK = (K <= 64) ? K : 64;

    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(2);

    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, /*transpose=*/false);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, /*transpose=*/false);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);

    cubeTiling.SetFixSplit(baseM, baseN, baseK);
    cubeTiling.SetBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    MatmulWithSmallKDimensionCustomTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    // Keep stable keying: key=2 enables optional UB tmp workspace in kernel.
    // Use platform branching only for key selection, not for op registration.
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        context->SetBlockDim(2);
        context->SetTilingKey(2);
    } else {
        context->SetBlockDim(1);
        context->SetTilingKey(1);
    }

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class MatmulWithSmallKDimensionCustom : public OpDef {
public:
    explicit MatmulWithSmallKDimensionCustom(const char *name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            // Register multiple typical configs to avoid "tiling branch unreachable" pitfalls.
            .AddConfig("ascend910b")
            .AddConfig("ascend910")
            .AddConfig("ascend310p");
    }
};

OP_ADD(MatmulWithSmallKDimensionCustom);

} // namespace ops
