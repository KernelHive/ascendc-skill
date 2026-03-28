
#include "gemm_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace optiling {
/**
 * GEMMCustom: y = x @ weight_t
 *   x        : (M,K) BF16 ND contiguous
 *   weight_t : (K,N) BF16 ND contiguous
 *   y        : (M,N) BF16 ND contiguous
 */
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto shape_x = context->GetInputTensor(0)->GetOriginShape();
    auto shape_wt = context->GetInputTensor(1)->GetOriginShape();

    if (shape_x.GetDimNum() != 2 || shape_wt.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    int32_t M = shape_x.GetDim(0);
    int32_t K = shape_x.GetDim(1);
    int32_t K2 = shape_wt.GetDim(0);
    int32_t N = shape_wt.GetDim(1);
    if (K != K2 || M <= 0 || N <= 0 || K <= 0) {
        return ge::GRAPH_FAILED;
    }

    int32_t baseM = 128;
    int32_t baseN = 128;

    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(2);
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);
    cubeTiling.SetFixSplit(baseM, baseN, -1);
    cubeTiling.SetBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    GEMMCustomTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    // Follow the known-good template: fixed small blockDim + tilingKey gate for tmp UB usage.
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
class GEMMCustom : public OpDef {
public:
    explicit GEMMCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Input("weight_t")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(GEMMCustom);
} // namespace ops
