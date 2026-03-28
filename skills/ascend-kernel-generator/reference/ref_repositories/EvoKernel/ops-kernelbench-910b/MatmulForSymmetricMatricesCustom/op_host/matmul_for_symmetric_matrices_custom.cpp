
#include "matmul_for_symmetric_matrices_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto shape_a = context->GetInputTensor(0)->GetOriginShape();
    auto shape_b = context->GetInputTensor(1)->GetOriginShape();

    // Expect A: (N,N), B: (N,N)
    if (shape_a.GetDimNum() != 2 || shape_b.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int32_t M = static_cast<int32_t>(shape_a.GetDim(0));
    const int32_t K = static_cast<int32_t>(shape_a.GetDim(1));
    const int32_t Kb = static_cast<int32_t>(shape_b.GetDim(0));
    const int32_t N = static_cast<int32_t>(shape_b.GetDim(1));

    if (M <= 0 || N <= 0 || K <= 0) {
        return ge::GRAPH_FAILED;
    }
    // Matmul constraints: (M,K) x (K,N)
    if (K != Kb) {
        return ge::GRAPH_FAILED;
    }
    // This custom op is intended for symmetric square matrices, but kernel computes general matmul.
    // Enforce square inputs to match model contract and keep tiling simple.
    if (M != K || K != N) {
        return ge::GRAPH_FAILED;
    }

    // Good baseline for 4096: moderate tiles; let tiler decide internal K blocking.
    const int32_t baseM = 128;
    const int32_t baseN = 128;

    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(2);

    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);

    cubeTiling.SetFixSplit(baseM, baseN, -1);
    cubeTiling.SetBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    MatmulForSymmetricMatricesCustomTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    // Mirror the known-good dual-key pattern: some SoCs need VECCALC tmp UB for mmformat.
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

class MatmulForSymmetricMatricesCustom : public OpDef {
public:
    explicit MatmulForSymmetricMatricesCustom(const char *name) : OpDef(name)
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
            .AddConfig("ascend910b");
    }
};

OP_ADD(MatmulForSymmetricMatricesCustom);

} // namespace ops
