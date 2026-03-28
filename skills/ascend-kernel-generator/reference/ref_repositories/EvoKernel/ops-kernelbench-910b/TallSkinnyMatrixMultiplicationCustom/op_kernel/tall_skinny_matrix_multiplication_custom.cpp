
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

template <typename aType, typename bType, typename cType>
class TallSkinnyMatmulKernel {
public:
    __aicore__ inline TallSkinnyMatmulKernel() {}

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
__aicore__ inline void TallSkinnyMatmulKernel<aType, bType, cType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling)
{
    (void)workspace;
    this->tiling = tiling;

    aBase.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), static_cast<uint64_t>(tiling.M) * tiling.Ka);
    bBase.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), static_cast<uint64_t>(tiling.Kb) * tiling.N);
    cBase.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), static_cast<uint64_t>(tiling.M) * tiling.N);

    if (GetSysWorkSpacePtr() == nullptr) {
        ok = false;
    }
}

template <typename aType, typename bType, typename cType>
template <bool setTmpSpace>
__aicore__ inline void TallSkinnyMatmulKernel<aType, bType, cType>::Process(AscendC::TPipe *pipe)
{
    if (!ok) return;

    if constexpr (setTmpSpace) {
        AscendC::TBuf<> tmpMMFormatUb;
        pipe->InitBuffer(tmpMMFormatUb, AscendC::TOTAL_VEC_LOCAL_SIZE);
        AscendC::LocalTensor<uint8_t> mmformatUb = tmpMMFormatUb.Get<uint8_t>(AscendC::TOTAL_VEC_LOCAL_SIZE);
        matmulObj.SetLocalWorkspace(mmformatUb);
    }

    // Use tiler-chosen tile sizes.
    uint32_t scM = static_cast<uint32_t>(tiling.singleCoreM);
    uint32_t scN = static_cast<uint32_t>(tiling.singleCoreN);
    if (scM == 0) scM = 128;
    if (scN == 0) scN = 32;

    // Group multiple N tiles per (mTile,nGroup) assignment; MUST be disjoint.
    constexpr uint32_t kNGroup = 4;

    const uint32_t M  = static_cast<uint32_t>(tiling.M);
    const uint32_t N  = static_cast<uint32_t>(tiling.N);
    const uint32_t Ka = static_cast<uint32_t>(tiling.Ka);

    const uint32_t mTiles  = CeilDivU32(M, scM);
    const uint32_t nTiles  = CeilDivU32(N, scN);
    const uint32_t nGroups = CeilDivU32(nTiles, kNGroup);
    if (mTiles == 0 || nTiles == 0 || nGroups == 0) return;

    const uint32_t bid  = static_cast<uint32_t>(GetBlockIdx());
    uint32_t bdim = static_cast<uint32_t>(GetBlockNum());
    if (bdim == 0) bdim = 1;

    // Total groups across the 2D tile space, mapped linearly and grid-stride by blocks.
    const uint32_t totalGroups = mTiles * nGroups;

    for (uint32_t groupLinear = bid; groupLinear < totalGroups; groupLinear += bdim) {
        // Map linear group -> (mIdx, nGroupIdx) in nGroup-major order to keep B/C contiguous across the inner loop.
        const uint32_t nGroupIdx = groupLinear % nGroups;
        const uint32_t mIdx      = groupLinear / nGroups;

        const uint32_t m0 = mIdx * scM;
        if (m0 >= M) continue;

        const uint32_t nTileStart = nGroupIdx * kNGroup;
        if (nTileStart >= nTiles) continue;

        // Compute A base once for this M tile.
        AscendC::GlobalTensor<aType> aGlobalBase = aBase[static_cast<uint64_t>(m0) * Ka];

        // Process up to kNGroup disjoint N-tiles for this group.
        uint32_t maxGroup = nTiles - nTileStart;
        if (maxGroup > kNGroup) maxGroup = kNGroup;

        #pragma unroll
        for (uint32_t g = 0; g < kNGroup; ++g) {
            if (g >= maxGroup) break;
            const uint32_t nTile = nTileStart + g;
            const uint32_t n0 = nTile * scN;
            if (n0 >= N) break;

            AscendC::GlobalTensor<bType> bGlobal = bBase[static_cast<uint64_t>(n0)];
            AscendC::GlobalTensor<cType> cGlobal = cBase[static_cast<uint64_t>(m0) * N + static_cast<uint64_t>(n0)];

            matmulObj.SetTensorA(aGlobalBase);
            matmulObj.SetTensorB(bGlobal);
            matmulObj.IterateAll(cGlobal);
            matmulObj.End();
        }
    }
}

extern "C" __global__ __aicore__ void tall_skinny_matrix_multiplication_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TallSkinnyMatmulKernel<float, float, float> matmulKernel;
    AscendC::TPipe pipe;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &tilingData.cubeTilingData);
    matmulKernel.Init(a, b, c, workspace, tilingData.cubeTilingData);

    if (TILING_KEY_IS(2)) {
        matmulKernel.Process<true>(&pipe);
    } else {
        matmulKernel.Process(&pipe);
    }
}
