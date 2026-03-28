
#include "batched_matrix_multiplication_custom_tiling.h"
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
    if (shape_a.GetDimNum() != 3 || shape_b.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const int32_t batchA = static_cast<int32_t>(shape_a.GetDim(0));
    const int32_t M      = static_cast<int32_t>(shape_a.GetDim(1));
    const int32_t K      = static_cast<int32_t>(shape_a.GetDim(2));

    const int32_t batchB = static_cast<int32_t>(shape_b.GetDim(0));
    const int32_t Kb     = static_cast<int32_t>(shape_b.GetDim(1));
    const int32_t N      = static_cast<int32_t>(shape_b.GetDim(2));

    if (batchA <= 0 || M <= 0 || K <= 0 || N <= 0) return ge::GRAPH_FAILED;
    if (batchB != batchA || Kb != K) return ge::GRAPH_FAILED;

    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(2);

    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);

    // Keep conservative split.
    const int32_t baseM = 128;
    const int32_t baseN = 256;
    cubeTiling.SetFixSplit(baseM, baseN, -1);

    cubeTiling.SetBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    BatchedMatrixMultiplicationCustomTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    tiling.set_batch(static_cast<uint32_t>(batchA));

    // Group a few adjacent N-tiles per block to amortize API overhead.
    // Keep small to preserve parallelism.
    uint32_t tilesPerBlockN = 2;
    tiling.set_tilesPerBlockN(tilesPerBlockN);

    uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNum());
    if (coreNum == 0U) coreNum = 1U;

    // Stable launch: primarily batch-parallel, with mild oversubscription to hide gaps.
    // This avoids tying launch size to M/N tile counts (which can create unstable patterns).
    uint32_t blockDim = static_cast<uint32_t>(batchA);
    if (blockDim < coreNum) blockDim = coreNum;
    uint32_t cap = coreNum * 2U;
    if (blockDim > cap) blockDim = cap;
    if (blockDim == 0U) blockDim = 1U;
    context->SetBlockDim(blockDim);

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

class BatchedMatrixMultiplicationCustom : public OpDef {
public:
    explicit BatchedMatrixMultiplicationCustom(const char *name) : OpDef(name)
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

OP_ADD(BatchedMatrixMultiplicationCustom);

} // namespace ops
