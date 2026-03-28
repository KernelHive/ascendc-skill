
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

template <typename aType, typename bType, typename cType>
class BmmKernelNd {
public:
    __aicore__ inline BmmKernelNd() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace,
                               const TCubeTiling &tiling_, uint32_t batch_, uint32_t tilesPerBlockN_)
    {
        (void)workspace;
        tiling = tiling_;
        batch = batch_;
        tilesPerBlockN = tilesPerBlockN_ == 0U ? 1U : tilesPerBlockN_;

        const uint64_t M  = static_cast<uint64_t>(tiling.M);
        const uint64_t N  = static_cast<uint64_t>(tiling.N);
        const uint64_t Ka = static_cast<uint64_t>(tiling.Ka);
        const uint64_t Kb = static_cast<uint64_t>(tiling.Kb);

        perA = M * Ka;
        perB = Kb * N;
        perC = M * N;

        aBase.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), static_cast<uint64_t>(batch) * perA);
        bBase.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), static_cast<uint64_t>(batch) * perB);
        cBase.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), static_cast<uint64_t>(batch) * perC);

        ok = (GetSysWorkSpacePtr() != nullptr);
    }

    template <bool setTmpSpace = false>
    __aicore__ inline void Process(AscendC::TPipe *pipe)
    {
        if (!ok) return;

        if constexpr (setTmpSpace) {
            AscendC::TBuf<> tmpMMFormatUb;
            pipe->InitBuffer(tmpMMFormatUb, AscendC::TOTAL_VEC_LOCAL_SIZE);
            AscendC::LocalTensor<uint8_t> mmformatUb = tmpMMFormatUb.Get<uint8_t>(AscendC::TOTAL_VEC_LOCAL_SIZE);
            matmulObj.SetLocalWorkspace(mmformatUb);
        }

        const uint32_t M = static_cast<uint32_t>(tiling.M);
        const uint32_t N = static_cast<uint32_t>(tiling.N);
        const uint32_t K = static_cast<uint32_t>(tiling.Ka);

        uint32_t scM = static_cast<uint32_t>(tiling.singleCoreM);
        uint32_t scN = static_cast<uint32_t>(tiling.singleCoreN);
        if (scM == 0U) scM = 128U;
        if (scN == 0U) scN = 256U;

        const uint32_t mTiles = CeilDivU32(M, scM);
        const uint32_t nTiles = CeilDivU32(N, scN);
        if (mTiles == 0U || nTiles == 0U || batch == 0U) return;

        const uint32_t bid = static_cast<uint32_t>(GetBlockIdx());
        uint32_t bdim = static_cast<uint32_t>(GetBlockNum());
        if (bdim == 0U) bdim = 1U;

        // Batch-first mapping: each block processes batches in a strided pattern.
        for (uint32_t b = bid; b < batch; b += bdim) {
            AscendC::GlobalTensor<aType> aG0 = aBase[static_cast<uint64_t>(b) * perA];
            AscendC::GlobalTensor<bType> bG0 = bBase[static_cast<uint64_t>(b) * perB];
            AscendC::GlobalTensor<cType> cG0 = cBase[static_cast<uint64_t>(b) * perC];

            // Iterate tiles in (mTile, nTile) order, grouping a few consecutive N tiles.
            for (uint32_t mTile = 0; mTile < mTiles; ++mTile) {
                const uint32_t m0 = mTile * scM;
                if (m0 >= M) break;

                // Set A once per mTile.
                AscendC::GlobalTensor<aType> aG = aG0[static_cast<uint64_t>(m0) * static_cast<uint64_t>(K)];
                matmulObj.SetTensorA(aG, false);

                for (uint32_t nTile0 = 0; nTile0 < nTiles; nTile0 += tilesPerBlockN) {
                    // Process up to tilesPerBlockN adjacent N tiles.
                    const uint32_t nEnd = (nTile0 + tilesPerBlockN < nTiles) ? (nTile0 + tilesPerBlockN) : nTiles;
                    for (uint32_t nTile = nTile0; nTile < nEnd; ++nTile) {
                        const uint32_t n0 = nTile * scN;
                        if (n0 >= N) break;

                        AscendC::GlobalTensor<bType> bG = bG0[static_cast<uint64_t>(n0)];
                        matmulObj.SetTensorB(bG, false);

                        AscendC::GlobalTensor<cType> cG =
                            cG0[static_cast<uint64_t>(m0) * static_cast<uint64_t>(N) + static_cast<uint64_t>(n0)];

                        int32_t useM = static_cast<int32_t>(scM);
                        int32_t useN = static_cast<int32_t>(scN);
                        const int32_t iM0 = static_cast<int32_t>(m0);
                        const int32_t iN0 = static_cast<int32_t>(n0);
                        const int32_t iM  = static_cast<int32_t>(M);
                        const int32_t iN  = static_cast<int32_t>(N);

                        if (iM0 + useM > iM) useM = iM - iM0;
                        if (iN0 + useN > iN) useN = iN - iN0;
                        if (useM <= 0 || useN <= 0) continue;

                        matmulObj.SetSingleShape(useM, useN, static_cast<int32_t>(K));
                        matmulObj.IterateAll(cG);
                        matmulObj.End();
                    }
                }
            }
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
    TCubeTiling tiling;
    uint32_t batch = 0;
    uint32_t tilesPerBlockN = 1;
    uint64_t perA = 0, perB = 0, perC = 0;
    bool ok {true};
};

extern "C" __global__ __aicore__ void batched_matrix_multiplication_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    BmmKernelNd<float, float, float> op;
    AscendC::TPipe pipe;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj, &tilingData.cubeTilingData);
    op.Init(a, b, c, workspace, tilingData.cubeTilingData, tilingData.batch, tilingData.tilesPerBlockN);

    if (TILING_KEY_IS(2)) {
        op.Process<true>(&pipe);
    } else {
        op.Process(&pipe);
    }
}
