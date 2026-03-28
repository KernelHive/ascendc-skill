
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }
__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) { return (a < b) ? a : b; }

template <typename aType, typename bType, typename cType>
class MatmulKernelTransposedA {
public:
    __aicore__ inline MatmulKernelTransposedA() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace,
                               const TCubeTiling &tilingIn, uint32_t nGroupIn);

    template <bool setTmpSpace = false>
    __aicore__ inline void Process(AscendC::TPipe *pipe);

public:
    Matmul<
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>
    > matmulObj;

private:
    AscendC::GlobalTensor<aType> aBase; // storage: (K, M), treated transposed as (M, K)
    AscendC::GlobalTensor<bType> bBase; // storage: (K, N)
    AscendC::GlobalTensor<cType> cBase; // storage: (M, N)

    TCubeTiling tiling;

    uint32_t M = 0, N = 0, K = 0;
    uint32_t scM = 0, scN = 0;
    uint32_t mTiles = 0, nTiles = 0;
    uint32_t nGroup = 1;

    bool ok {true};
};

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernelTransposedA<aType, bType, cType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tilingIn, uint32_t nGroupIn)
{
    (void)workspace;
    tiling = tilingIn;

    M = static_cast<uint32_t>(tiling.M);
    N = static_cast<uint32_t>(tiling.N);
    K = static_cast<uint32_t>(tiling.Ka);

    scM = static_cast<uint32_t>(tiling.singleCoreM);
    scN = static_cast<uint32_t>(tiling.singleCoreN);
    if (scM == 0U) scM = 128U;
    if (scN == 0U) scN = 128U;

    mTiles = CeilDivU32(M, scM);
    nTiles = CeilDivU32(N, scN);

    nGroup = (nGroupIn == 0U) ? 1U : nGroupIn;
    if (nGroup > nTiles) nGroup = nTiles;
    if (nGroup == 0U) nGroup = 1U;

    aBase.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), static_cast<uint64_t>(tiling.Ka) * tiling.M);
    bBase.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), static_cast<uint64_t>(tiling.Kb) * tiling.N);
    cBase.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), static_cast<uint64_t>(tiling.M) * tiling.N);

    ok = (GetSysWorkSpacePtr() != nullptr) && (mTiles != 0U) && (nTiles != 0U) && (K != 0U);
}

template <typename aType, typename bType, typename cType>
template <bool setTmpSpace>
__aicore__ inline void MatmulKernelTransposedA<aType, bType, cType>::Process(AscendC::TPipe *pipe)
{
    if (!ok) return;

    if constexpr (setTmpSpace) {
        AscendC::TBuf<> tmpMMFormatUb;
        pipe->InitBuffer(tmpMMFormatUb, AscendC::TOTAL_VEC_LOCAL_SIZE);
        AscendC::LocalTensor<uint8_t> mmformatUb = tmpMMFormatUb.Get<uint8_t>(AscendC::TOTAL_VEC_LOCAL_SIZE);
        matmulObj.SetLocalWorkspace(mmformatUb);
    }

    const uint32_t bid = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t bdim = static_cast<uint32_t>(GetBlockNum());
    if (bdim == 0U) return;

    // Linearize tiles and assign contiguous groups to each block (grid-stride over tile groups).
    // Each group processes up to nGroup consecutive N-tiles for one M-tile to amortize control overhead.
    const uint32_t groupsPerM = CeilDivU32(nTiles, nGroup);
    const uint32_t totalGroups = mTiles * groupsPerM;

    for (uint32_t g = bid; g < totalGroups; g += bdim) {
        const uint32_t mt = (groupsPerM == 0U) ? 0U : (g / groupsPerM);
        const uint32_t ng = (groupsPerM == 0U) ? 0U : (g - mt * groupsPerM);

        const uint32_t m0u = mt * scM;
        if (m0u >= M) continue;
        int32_t m0 = static_cast<int32_t>(m0u);
        int32_t useM = static_cast<int32_t>(scM);
        if (static_cast<uint32_t>(m0 + useM) > M) useM = static_cast<int32_t>(M) - m0;
        if (useM <= 0) continue;

        // A is transposed: tile starts at "row m0" in logical (M,K) => base offset m0.
        AscendC::GlobalTensor<aType> aG = aBase[static_cast<uint64_t>(m0u)];

        // Process consecutive N tiles for this m0.
        const uint32_t ntStart = ng * nGroup;
        const uint32_t ntEnd = MinU32(ntStart + nGroup, nTiles);

        matmulObj.SetTensorA(aG, /*isTransposeA=*/true);

        for (uint32_t nt = ntStart; nt < ntEnd; ++nt) {
            const uint32_t n0u = nt * scN;
            if (n0u >= N) break;

            int32_t n0 = static_cast<int32_t>(n0u);
            int32_t useN = static_cast<int32_t>(scN);
            if (static_cast<uint32_t>(n0 + useN) > N) useN = static_cast<int32_t>(N) - n0;
            if (useN <= 0) continue;

            // B tile base at column n0 => offset n0 in row-major (K,N) storage.
            AscendC::GlobalTensor<bType> bG = bBase[static_cast<uint64_t>(n0u)];
            // C tile base at (m0,n0) => m0*N + n0
            AscendC::GlobalTensor<cType> cG =
                cBase[static_cast<uint64_t>(m0u) * static_cast<uint64_t>(N) + static_cast<uint64_t>(n0u)];

            matmulObj.SetTensorB(bG, /*isTransposeB=*/false);
            matmulObj.SetSingleShape(useM, useN, static_cast<int32_t>(K));
            matmulObj.IterateAll(cG);
            matmulObj.End();
        }
    }
}

extern "C" __global__ __aicore__ void matmul_with_transposed_a_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    MatmulKernelTransposedA<float, float, float> op;
    AscendC::TPipe pipe;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj, &tilingData.cubeTilingData);
    op.Init(a, b, c, workspace, tilingData.cubeTilingData, tilingData.nGroup);

    if (TILING_KEY_IS(2)) {
        op.Process<true>(&pipe);
    } else {
        op.Process(&pipe);
    }
}
