
#include "tall_skinny_matrix_multiplication_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace optiling {

static inline int32_t ClampI32(int32_t v, int32_t lo, int32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}
static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto shape_a = context->GetInputTensor(0)->GetOriginShape(); // A: (M, K)
    auto shape_b = context->GetInputTensor(1)->GetOriginShape(); // B: (K, N)

    if (shape_a.GetDimNum() != 2 || shape_b.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int32_t M = static_cast<int32_t>(shape_a.GetDim(0));
    const int32_t K = static_cast<int32_t>(shape_a.GetDim(1));
    const int32_t Kb = static_cast<int32_t>(shape_b.GetDim(0));
    const int32_t N = static_cast<int32_t>(shape_b.GetDim(1));
    if (M <= 0 || N <= 0 || K <= 0 || K != Kb) {
        return ge::GRAPH_FAILED;
    }

    // Tall/skinny heuristics:
    // - Make M tile large to amortize setup.
    // - Keep N tile modest but aligned (multiple of 16) to be cube-friendly.
    int32_t baseM = 256;
    if (M <= 4096) baseM = 128;
    if (M <= 2048) baseM = 64;
    baseM = ClampI32(baseM, 16, M);

    int32_t baseN = 32;
    if (N <= 64) baseN = 32;
    if (N <= 32) baseN = 16;
    if (N >= 256) baseN = 64;
    if (N >= 1024) baseN = 128;
    baseN = ClampI32(baseN, 16, N);

    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(2);

    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);

    cubeTiling.SetAlignSplit(16, 8, 16);
    cubeTiling.SetFixSplit(baseM, baseN, -1);

    cubeTiling.SetBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    TallSkinnyMatrixMultiplicationCustomTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    // Block mapping matches kernel grouping over N tiles.
    // We group kNGroup N-tiles together, so the number of independent groups is smaller and stable.
    constexpr uint32_t kNGroup = 4;
    uint32_t mTiles = CeilDivU32(static_cast<uint32_t>(M), static_cast<uint32_t>(baseM));
    uint32_t nTiles = CeilDivU32(static_cast<uint32_t>(N), static_cast<uint32_t>(baseN));
    uint32_t nGroups = CeilDivU32(nTiles, kNGroup);
    uint32_t totalGroups = mTiles * nGroups;
    if (totalGroups == 0) totalGroups = 1;

    uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNum());
    if (coreNum == 0) coreNum = 1;

    uint32_t blockDim = (totalGroups < coreNum) ? totalGroups : coreNum;
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    // Keep UB tmp workspace path for matmul interface stability.
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

class TallSkinnyMatrixMultiplicationCustom : public OpDef {
public:
    explicit TallSkinnyMatrixMultiplicationCustom(const char *name) : OpDef(name)
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

OP_ADD(TallSkinnyMatrixMultiplicationCustom);

} // namespace ops
