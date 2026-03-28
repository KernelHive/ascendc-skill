
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

template <typename aType, typename bType, typename cType>
class MatmulKernelTransposedB {
public:
    __aicore__ inline MatmulKernelTransposedB() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling);

    template <bool setTmpSpace = false>
    __aicore__ inline void Process(AscendC::TPipe *pipe);

public:
    Matmul<
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>
    > matmulObj;

    AscendC::GlobalTensor<aType> aBase;
    AscendC::GlobalTensor<bType> bBase;
    AscendC::GlobalTensor<cType> cBase;

    TCubeTiling tiling;
    bool ok {true};
};

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernelTransposedB<aType, bType, cType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling)
{
    (void)workspace;
    this->tiling = tiling;

    aBase.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), static_cast<uint64_t>(tiling.M) * tiling.Ka);
    bBase.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), static_cast<uint64_t>(tiling.N) * tiling.Kb);
    cBase.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), static_cast<uint64_t>(tiling.M) * tiling.N);

    if (GetSysWorkSpacePtr() == nullptr) {
        ok = false;
    }
}

template <typename aType, typename bType, typename cType>
template <bool setTmpSpace>
__aicore__ inline void MatmulKernelTransposedB<aType, bType, cType>::Process(AscendC::TPipe *pipe)
{
    if (!ok) return;

    if constexpr (setTmpSpace) {
        AscendC::TBuf<> tmpMMFormatUb;
        pipe->InitBuffer(tmpMMFormatUb, AscendC::TOTAL_VEC_LOCAL_SIZE);
        AscendC::LocalTensor<uint8_t> mmformatUb = tmpMMFormatUb.Get<uint8_t>(AscendC::TOTAL_VEC_LOCAL_SIZE);
        matmulObj.SetLocalWorkspace(mmformatUb);
    }

    // Hoist frequently used fields to reduce scalar/icache pressure.
    const uint32_t M  = static_cast<uint32_t>(tiling.M);
    const uint32_t N  = static_cast<uint32_t>(tiling.N);
    const uint32_t Ka = static_cast<uint32_t>(tiling.Ka);
    const uint32_t Kb = static_cast<uint32_t>(tiling.Kb);

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

    // Adaptive chunk size: small enough to keep periodic End() flush points, large enough for B reuse.
    uint32_t chunk = 8;
    if (mTiles <= 8) chunk = 4;
    if (mTiles <= 4) chunk = 2;
    if (chunk == 0) chunk = 1;

    // N-major mapping: each block owns a grid-stride subset of N tiles (better B temporal locality).
    for (uint32_t nTile = bid; nTile < nTiles; nTile += bdim) {
        const uint32_t n0 = nTile * scN;
        if (n0 >= N) continue;

        // Set B once per N tile and reuse across M-chunks.
        AscendC::GlobalTensor<bType> bGlobal = bBase[static_cast<uint64_t>(n0) * static_cast<uint64_t>(Kb)];
        matmulObj.SetTensorB(bGlobal, /*isTransposeB=*/true);

        // Process M tiles in chunks; End() per chunk to reduce long pipeline bubbles.
        for (uint32_t mStart = 0; mStart < mTiles; mStart += chunk) {
            const uint32_t mEnd = (mStart + chunk < mTiles) ? (mStart + chunk) : mTiles;

            // Running offsets to avoid repeated 64-bit multiplies in the inner loop.
            uint32_t m0 = mStart * scM;
            uint64_t aOff = static_cast<uint64_t>(m0) * static_cast<uint64_t>(Ka);
            uint64_t cOff = static_cast<uint64_t>(m0) * static_cast<uint64_t>(N) + static_cast<uint64_t>(n0);

            for (uint32_t mTile = mStart; mTile < mEnd; ++mTile) {
                if (m0 >= M) break;

                AscendC::GlobalTensor<aType> aGlobal = aBase[aOff];
                AscendC::GlobalTensor<cType> cGlobal = cBase[cOff];

                matmulObj.SetTensorA(aGlobal, /*isTransposeA=*/false);
                matmulObj.IterateAll(cGlobal);

                m0 += scM;
                aOff += static_cast<uint64_t>(scM) * static_cast<uint64_t>(Ka);
                cOff += static_cast<uint64_t>(scM) * static_cast<uint64_t>(N);
            }
            matmulObj.End();
        }
    }
}

extern "C" __global__ __aicore__ void matmul_with_transposed_b_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    MatmulKernelTransposedB<float, float, float> matmulKernel;
    AscendC::TPipe pipe;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &tilingData.cubeTilingData);
    matmulKernel.Init(a, b, c, workspace, tilingData.cubeTilingData);

    if (TILING_KEY_IS(2)) {
        matmulKernel.Process<true>(&pipe);
    } else {
        matmulKernel.Process(&pipe);
    }
}
