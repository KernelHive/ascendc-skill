
#include "matmul_with_irregular_shapes_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }
static inline uint32_t ClampU32(uint32_t x, uint32_t lo, uint32_t hi) {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    auto shape_a = context->GetInputTensor(0)->GetOriginShape(); // A: (M, K)
    auto shape_b = context->GetInputTensor(1)->GetOriginShape(); // B: (K, N)

    if (shape_a.GetDimNum() != 2 || shape_b.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int32_t M  = static_cast<int32_t>(shape_a.GetDim(0));
    const int32_t Ka = static_cast<int32_t>(shape_a.GetDim(1));
    const int32_t Kb = static_cast<int32_t>(shape_b.GetDim(0));
    const int32_t N  = static_cast<int32_t>(shape_b.GetDim(1));

    if (M <= 0 || N <= 0 || Ka <= 0 || Kb <= 0 || Ka != Kb) {
        return ge::GRAPH_FAILED;
    }
    const int32_t K = Ka;

    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(2);

    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, /*transpose=*/false);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, /*transpose=*/false);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);

    // Keep the known-good split to control tile-count for irregular N.
    const int32_t baseM = 128;
    const int32_t baseN = 192;
    cubeTiling.SetFixSplit(baseM, baseN, -1);
    cubeTiling.SetBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    MatmulWithIrregularShapesCustomTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNum());
    if (coreNum == 0) coreNum = 1;

    const uint32_t mTiles = CeilDivU32(static_cast<uint32_t>(M), static_cast<uint32_t>(baseM));
    const uint32_t nTiles = CeilDivU32(static_cast<uint32_t>(N), static_cast<uint32_t>(baseN));
    uint32_t totalTiles = mTiles * nTiles;
    if (totalTiles == 0) totalTiles = 1;

    // More concurrency to hide MTE2, but stay conservative for stability.
    uint32_t want = (totalTiles < coreNum) ? totalTiles : coreNum;
    // 910B is typically fine with moderate blockDim; cap at 16 to improve overlap without going extreme.
    uint32_t blockDim = ClampU32(want, 1U, 16U);
    context->SetBlockDim(blockDim);

    // Use matmul-intf path with explicit per-block UB tmp workspace.
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

class MatmulWithIrregularShapesCustom : public OpDef {
public:
    explicit MatmulWithIrregularShapesCustom(const char *name) : OpDef(name)
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

OP_ADD(MatmulWithIrregularShapesCustom);

} // namespace ops
