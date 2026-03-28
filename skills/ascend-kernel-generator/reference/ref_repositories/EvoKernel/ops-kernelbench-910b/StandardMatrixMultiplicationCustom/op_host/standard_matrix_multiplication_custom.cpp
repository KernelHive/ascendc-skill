
#include "standard_matrix_multiplication_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace optiling {

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    auto shape_a = context->GetInputTensor(0)->GetOriginShape();
    auto shape_b = context->GetInputTensor(1)->GetOriginShape();
    if (shape_a.GetDimNum() != 2 || shape_b.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int32_t M  = static_cast<int32_t>(shape_a.GetDim(0));
    const int32_t K  = static_cast<int32_t>(shape_a.GetDim(1));
    const int32_t Kb = static_cast<int32_t>(shape_b.GetDim(0));
    const int32_t N  = static_cast<int32_t>(shape_b.GetDim(1));
    if (M <= 0 || N <= 0 || K <= 0 || Kb != K) {
        return ge::GRAPH_FAILED;
    }

    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(2);

    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);

    // Keep stable, conservative hints (avoid relying on hint tuning for perf).
    const int32_t hintM = 128;
    const int32_t hintN = 128;
    cubeTiling.SetFixSplit(hintM, hintN, -1);
    cubeTiling.SetBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    StandardMatrixMultiplicationCustomTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    // Stable blockDim estimation based on hints only (avoid host/kernel mismatch).
    uint32_t mTiles = CeilDivU32(static_cast<uint32_t>(M), static_cast<uint32_t>(hintM));
    uint32_t nTiles = CeilDivU32(static_cast<uint32_t>(N), static_cast<uint32_t>(hintN));
    uint32_t totalTiles = mTiles * nTiles;
    if (totalTiles == 0) totalTiles = 1;

    uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNum());
    if (coreNum == 0) coreNum = 1;

    uint32_t blockDim = (totalTiles < coreNum) ? totalTiles : coreNum;
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    // Use tmp-UB path for mmformat workspace stability.
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

class StandardMatrixMultiplicationCustom : public OpDef {
public:
    explicit StandardMatrixMultiplicationCustom(const char *name) : OpDef(name)
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

OP_ADD(StandardMatrixMultiplicationCustom);

} // namespace ops
