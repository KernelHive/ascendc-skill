
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

template <typename aType, typename bType, typename cType>
class MatmulKernelIrregularOpt {
public:
    __aicore__ inline MatmulKernelIrregularOpt() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, const TCubeTiling &t)
    {
        tiling = t;

        const uint64_t M = static_cast<uint64_t>(tiling.M);
        const uint64_t N = static_cast<uint64_t>(tiling.N);
        const uint64_t Ka = static_cast<uint64_t>(tiling.Ka);
        const uint64_t Kb = static_cast<uint64_t>(tiling.Kb);

        aBase.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), M * Ka);
        bBase.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), Kb * N);
        cBase.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), M * N);

        ok = (GetSysWorkSpacePtr() != nullptr);
    }

    __aicore__ inline void BindWorkspaceOnce(AscendC::TPipe *pipe)
    {
        // Bind matmul local UB workspace once per block to avoid internal format buffer churn.
        AscendC::TBuf<> tmpMMFormatUb;
        pipe->InitBuffer(tmpMMFormatUb, AscendC::TOTAL_VEC_LOCAL_SIZE);
        AscendC::LocalTensor<uint8_t> mmformatUb = tmpMMFormatUb.Get<uint8_t>(AscendC::TOTAL_VEC_LOCAL_SIZE);
        matmulObj.SetLocalWorkspace(mmformatUb);
    }

    __aicore__ inline void Process(AscendC::TPipe *pipe)
    {
        if (!ok) return;
        if (TILING_KEY_IS(2)) {
            BindWorkspaceOnce(pipe);
        }

        const uint32_t M = static_cast<uint32_t>(tiling.M);
        const uint32_t N = static_cast<uint32_t>(tiling.N);
        const uint32_t K = static_cast<uint32_t>(tiling.Ka);

        uint32_t scM = static_cast<uint32_t>(tiling.singleCoreM);
        uint32_t scN = static_cast<uint32_t>(tiling.singleCoreN);
        if (scM == 0) scM = 128;
        if (scN == 0) scN = 192;

        const uint32_t mTiles = CeilDivU32(M, scM);
        const uint32_t nTiles = CeilDivU32(N, scN);
        const uint32_t totalTiles = mTiles * nTiles;
        if (totalTiles == 0) return;

        const uint32_t bid = static_cast<uint32_t>(GetBlockIdx());
        uint32_t bdim = static_cast<uint32_t>(GetBlockNum());
        if (bdim == 0) bdim = 1;

        // Keep per-tile decode cheap and predictable.
        for (uint32_t tile = bid; tile < totalTiles; tile += bdim) {
            const uint32_t nIdx = tile / mTiles;
            const uint32_t mIdx = tile - nIdx * mTiles;

            const uint32_t m0 = mIdx * scM;
            const uint32_t n0 = nIdx * scN;
            if (m0 >= M || n0 >= N) continue;

            uint32_t useM = M - m0;
            uint32_t useN = N - n0;
            if (useM > scM) useM = scM;
            if (useN > scN) useN = scN;

            // Explicitly rebase A/B per tile (required for correctness with offsets).
            const uint64_t aOff = static_cast<uint64_t>(m0) * static_cast<uint64_t>(K);
            const uint64_t bOff = static_cast<uint64_t>(n0);
            const uint64_t cOff = static_cast<uint64_t>(m0) * static_cast<uint64_t>(N) + static_cast<uint64_t>(n0);

            AscendC::GlobalTensor<aType> aTile = aBase[aOff];
            AscendC::GlobalTensor<bType> bTile = bBase[bOff];
            AscendC::GlobalTensor<cType> cTile = cBase[cOff];

            // Minimal per-tile state change; keep lifecycle per tile (IterateAll + End).
            matmulObj.SetTensorA(aTile, /*isTransposeA=*/false);
            matmulObj.SetTensorB(bTile, /*isTransposeB=*/false);
            matmulObj.SetSingleShape(static_cast<int32_t>(useM),
                                     static_cast<int32_t>(useN),
                                     static_cast<int32_t>(K));
            matmulObj.IterateAll(cTile);
            matmulObj.End();
        }
    }

public:
    Matmul<
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>
    > matmulObj;

private:
    AscendC::GlobalTensor<aType> aBase;
    AscendC::GlobalTensor<bType> bBase;
    AscendC::GlobalTensor<cType> cBase;
    TCubeTiling tiling {};
    bool ok {true};
};

extern "C" __global__ __aicore__ void matmul_with_irregular_shapes_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);

    if (GetSysWorkSpacePtr() == nullptr) return;

    AscendC::TPipe pipe;
    MatmulKernelIrregularOpt<float, float, float> k;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), k.matmulObj, &tilingData.cubeTilingData);
    k.Init(a, b, c, tilingData.cubeTilingData);
    k.Process(&pipe);
}
