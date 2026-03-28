
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

template <typename aType, typename bType, typename cType>
class MatmulKernelTransposedBoth {
public:
    __aicore__ inline MatmulKernelTransposedBoth() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling);

    template <bool setTmpSpace = false>
    __aicore__ inline void Process(AscendC::TPipe *pipe);

public:
    Matmul<
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>
    > matmulObj;

    AscendC::GlobalTensor<aType> aBase; // stored A: (K, M)
    AscendC::GlobalTensor<bType> bBase; // stored B: (N, K)
    AscendC::GlobalTensor<cType> cBase; // stored C: (M, N)

    TCubeTiling tiling;
    bool ok {true};
};

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernelTransposedBoth<aType, bType, cType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling)
{
    (void)workspace;
    this->tiling = tiling;

    // tiling contains logical shape (M, N, K); stored buffers are A(K,M), B(N,K), C(M,N).
    aBase.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), static_cast<uint64_t>(tiling.Ka) * tiling.M); // K*M
    bBase.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), static_cast<uint64_t>(tiling.N) * tiling.Kb); // N*K
    cBase.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), static_cast<uint64_t>(tiling.M) * tiling.N);  // M*N

    if (GetSysWorkSpacePtr() == nullptr) {
        ok = false;
    }
}

template <typename aType, typename bType, typename cType>
template <bool setTmpSpace>
__aicore__ inline void MatmulKernelTransposedBoth<aType, bType, cType>::Process(AscendC::TPipe *pipe)
{
    if (!ok) return;

    if constexpr (setTmpSpace) {
        AscendC::TBuf<> tmpMMFormatUb;
        pipe->InitBuffer(tmpMMFormatUb, AscendC::TOTAL_VEC_LOCAL_SIZE);
        AscendC::LocalTensor<uint8_t> mmformatUb = tmpMMFormatUb.Get<uint8_t>(AscendC::TOTAL_VEC_LOCAL_SIZE);
        matmulObj.SetLocalWorkspace(mmformatUb);
    }

    const uint32_t M  = static_cast<uint32_t>(tiling.M);
    const uint32_t N  = static_cast<uint32_t>(tiling.N);
    const uint32_t Kb = static_cast<uint32_t>(tiling.Kb);

    // Use library-selected per-core tile sizes on device; fall back to safe defaults.
    uint32_t scM = static_cast<uint32_t>(tiling.singleCoreM);
    uint32_t scN = static_cast<uint32_t>(tiling.singleCoreN);
    if (scM == 0) scM = 128;
    if (scN == 0) scN = 128;

    const uint32_t mTiles = CeilDivU32(M, scM);
    const uint32_t nTiles = CeilDivU32(N, scN);
    if (mTiles == 0 || nTiles == 0) return;

    const uint32_t bid = static_cast<uint32_t>(GetBlockIdx());
    uint32_t bdim = static_cast<uint32_t>(GetBlockNum());
    if (bdim == 0) bdim = 1;

    // Pipeline optimization: map blocks over N-tiles and sweep all M-tiles per N-tile.
    // This reuses SetTensorB and calls End() once per N-tile, reducing sync/pipeline gaps.
    for (uint32_t nTile = bid; nTile < nTiles; nTile += bdim) {
        const uint32_t n0 = nTile * scN;
        if (n0 >= N) continue;

        // transposeB=true and stored B(N,K): base at row n0 => offset n0*K.
        AscendC::GlobalTensor<bType> bGlobal = bBase[static_cast<uint64_t>(n0) * static_cast<uint64_t>(Kb)];
        matmulObj.SetTensorB(bGlobal, /*isTransposeB=*/true);

        for (uint32_t mTile = 0; mTile < mTiles; ++mTile) {
            const uint32_t m0 = mTile * scM;
            if (m0 >= M) break;

            // transposeA=true and stored A(K,M): base at column m0 => offset m0.
            AscendC::GlobalTensor<aType> aGlobal = aBase[static_cast<uint64_t>(m0)];
            AscendC::GlobalTensor<cType> cGlobal =
                cBase[static_cast<uint64_t>(m0) * static_cast<uint64_t>(N) + static_cast<uint64_t>(n0)];

            matmulObj.SetTensorA(aGlobal, /*isTransposeA=*/true);
            matmulObj.IterateAll(cGlobal);
        }
        matmulObj.End();
    }
}

extern "C" __global__ __aicore__ void matmul_with_transposed_both_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    MatmulKernelTransposedBoth<float, float, float> matmulKernel;
    AscendC::TPipe pipe;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &tilingData.cubeTilingData);
    matmulKernel.Init(a, b, c, workspace, tilingData.cubeTilingData);

    if (TILING_KEY_IS(2)) {
        matmulKernel.Process<true>(&pipe);
    } else {
        matmulKernel.Process<false>(&pipe);
    }
}
