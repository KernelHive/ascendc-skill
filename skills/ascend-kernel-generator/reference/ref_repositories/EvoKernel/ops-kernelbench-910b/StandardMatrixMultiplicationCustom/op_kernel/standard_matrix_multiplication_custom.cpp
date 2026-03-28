
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

template <typename aType, typename bType, typename cType>
class MatmulKernelNN {
public:
    __aicore__ inline MatmulKernelNN() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling);

    template <bool setTmpSpace = false>
    __aicore__ inline void Process(AscendC::TPipe *pipe);

public:
    Matmul<
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>
    > matmulObj;

    __gm__ aType *aPtr {nullptr};
    __gm__ bType *bPtr {nullptr};
    __gm__ cType *cPtr {nullptr};

    TCubeTiling tiling;
    bool ok {true};
};

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernelNN<aType, bType, cType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling)
{
    (void)workspace;
    this->tiling = tiling;

    aPtr = reinterpret_cast<__gm__ aType *>(a);
    bPtr = reinterpret_cast<__gm__ bType *>(b);
    cPtr = reinterpret_cast<__gm__ cType *>(c);

    if (GetSysWorkSpacePtr() == nullptr) {
        ok = false;
    }
}

template <typename aType, typename bType, typename cType>
template <bool setTmpSpace>
__aicore__ inline void MatmulKernelNN<aType, bType, cType>::Process(AscendC::TPipe *pipe)
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
    const uint32_t Ka = static_cast<uint32_t>(tiling.Ka);

    uint32_t scM = static_cast<uint32_t>(tiling.singleCoreM);
    uint32_t scN = static_cast<uint32_t>(tiling.singleCoreN);
    if (scM == 0) scM = 128;
    if (scN == 0) scN = 128;

    const uint32_t mTiles = CeilDivU32(M, scM);
    const uint32_t nTiles = CeilDivU32(N, scN);
    if (mTiles == 0 || nTiles == 0) return;
    const uint32_t totalTiles = mTiles * nTiles;

    const uint32_t bid = static_cast<uint32_t>(GetBlockIdx());
    uint32_t bdim = static_cast<uint32_t>(GetBlockNum());
    if (bdim == 0) bdim = 1;

    // Use incremental address updates to reduce div/mod and 64-bit multiplies per iteration.
    // Keep End() per-tile (do NOT hoist) to respect matmul library lifecycle.
    for (uint32_t tid = bid; tid < totalTiles; tid += bdim) {
        const uint32_t mTile = tid / nTiles;
        const uint32_t nTile = tid - mTile * nTiles;

        const uint32_t m0 = mTile * scM;
        const uint32_t n0 = nTile * scN;
        if (m0 >= M || n0 >= N) continue;

        const uint64_t aOff = static_cast<uint64_t>(m0) * static_cast<uint64_t>(Ka);
        const uint64_t bOff = static_cast<uint64_t>(n0);
        const uint64_t cOff = static_cast<uint64_t>(m0) * static_cast<uint64_t>(N) + static_cast<uint64_t>(n0);

        AscendC::GlobalTensor<aType> aGlobal;
        AscendC::GlobalTensor<bType> bGlobal;
        AscendC::GlobalTensor<cType> cGlobal;

        // Create lightweight tensor views (avoid repeated base[] operator machinery).
        aGlobal.SetGlobalBuffer(aPtr + aOff, static_cast<uint64_t>(M - m0) * Ka);
        bGlobal.SetGlobalBuffer(bPtr + bOff, static_cast<uint64_t>(tiling.Kb) * N - bOff);
        cGlobal.SetGlobalBuffer(cPtr + cOff, static_cast<uint64_t>(M - m0) * N - n0);

        matmulObj.SetTensorA(aGlobal, false);
        matmulObj.SetTensorB(bGlobal, false);
        matmulObj.IterateAll(cGlobal);
        matmulObj.End();
    }
}

extern "C" __global__ __aicore__ void standard_matrix_multiplication_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    MatmulKernelNN<float, float, float> matmulKernel;
    AscendC::TPipe pipe;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &tilingData.cubeTilingData);
    matmulKernel.Init(a, b, c, workspace, tilingData.cubeTilingData);

    if (TILING_KEY_IS(2)) {
        matmulKernel.Process<true>(&pipe);
    } else {
        matmulKernel.Process(&pipe);
    }
}
