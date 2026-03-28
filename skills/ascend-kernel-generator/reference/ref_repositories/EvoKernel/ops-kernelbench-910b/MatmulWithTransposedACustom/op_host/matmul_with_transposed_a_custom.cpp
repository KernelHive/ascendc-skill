
#include "matmul_with_transposed_a_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    auto shape_a = context->GetInputTensor(0)->GetOriginShape(); // A: (K, M)
    auto shape_b = context->GetInputTensor(1)->GetOriginShape(); // B: (K, N)

    if (shape_a.GetDimNum() != 2 || shape_b.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int32_t K_a = static_cast<int32_t>(shape_a.GetDim(0));
    const int32_t M   = static_cast<int32_t>(shape_a.GetDim(1));
    const int32_t K_b = static_cast<int32_t>(shape_b.GetDim(0));
    const int32_t N   = static_cast<int32_t>(shape_b.GetDim(1));

    if (M <= 0 || N <= 0 || K_a <= 0 || K_a != K_b) {
        return ge::GRAPH_FAILED;
    }
    const int32_t K = K_a;
    (void)K;

    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(2);

    // A is logically transposed: storage (K,M) is treated as (M,K)
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, /*transpose=*/true);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, /*transpose=*/false);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);

    // Use moderate fixed splits; kernel will group tiles to reduce setup/pipeline gaps.
    const int32_t baseM = 128;
    const int32_t baseN = 128;
    cubeTiling.SetFixSplit(baseM, baseN, -1);
    cubeTiling.SetBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    MatmulWithTransposedACustomTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    // Choose a small N grouping factor to amortize per-tile setup while keeping enough parallelism.
    // Larger N benefits from slightly larger grouping.
    uint32_t nGroup = 2U;
    if (N >= 4096) nGroup = 4U;
    tiling.set_nGroup(nGroup);

    // Stable blockDim: a fraction of core count (avoid launching too few tiny blocks or too many).
    uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNum());
    if (coreNum == 0U) coreNum = 1U;

    // Guarantee at least some parallel blocks; cap to coreNum.
    uint32_t blockDim = coreNum;
    // If the problem is small in tiles, don't oversubscribe.
    uint32_t mTilesHint = CeilDivU32(static_cast<uint32_t>(M), static_cast<uint32_t>(baseM));
    uint32_t nTilesHint = CeilDivU32(static_cast<uint32_t>(N), static_cast<uint32_t>(baseN));
    uint32_t totalTilesHint = mTilesHint * nTilesHint;
    if (totalTilesHint == 0U) totalTilesHint = 1U;
    if (blockDim > totalTilesHint) blockDim = totalTilesHint;
    if (blockDim == 0U) blockDim = 1U;

    context->SetBlockDim(blockDim);

    // Keep UB temp workspace path for steadier matmul interface performance.
    context->SetTilingKey(2);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

class MatmulWithTransposedACustom : public OpDef {
public:
    explicit MatmulWithTransposedACustom(const char *name) : OpDef(name)
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

OP_ADD(MatmulWithTransposedACustom);

} // namespace ops
