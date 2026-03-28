
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return (a + b - 1U) / b;
}

template <typename aType, typename bType, typename cType>
class MatmulUpperTriKernel {
public:
    __aicore__ inline MatmulUpperTriKernel() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling);

    template <bool setTmpSpace = false>
    __aicore__ inline void Process(AscendC::TPipe *pipe);

private:
    struct TileInfo {
        int32_t rowStart;
        int32_t colStart;
        int32_t tileM;
        int32_t tileN;
        int32_t offsetA;
        int32_t offsetB;
        int32_t offsetC;
        bool valid;
    };

    __aicore__ inline void CalcTileInfo(int32_t blockIdx, const TCubeTiling &t, TileInfo &ti);

    __aicore__ inline void EnsureZeroSlab(AscendC::TPipe *pipe);

    __aicore__ inline void ZeroFullTileLocal(AscendC::GlobalTensor<cType> cTile, int32_t Nstride,
                                             int32_t tileM, int32_t tileN);

    __aicore__ inline void ZeroStrictLowerPrefixLocal(AscendC::GlobalTensor<cType> cTile, int32_t Nstride,
                                                      int32_t rowStart, int32_t colStart,
                                                      int32_t tileM, int32_t tileN);

public:
    Matmul<
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>
    > matmulObj;

    AscendC::GlobalTensor<aType> aGlobal; // (M,K)
    AscendC::GlobalTensor<bType> bGlobal; // (K,N)
    AscendC::GlobalTensor<cType> cGlobal; // (M,N)
    TCubeTiling tiling_;

    static constexpr int32_t ZERO_SLAB_ELEMS = 8192;
    AscendC::TBuf<> zeroBuf_;
    AscendC::LocalTensor<cType> zeroSlab_;
    bool zeroReady_ = false;
};

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulUpperTriKernel<aType, bType, cType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling)
{
    (void)workspace;
    tiling_ = tiling;

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), static_cast<uint64_t>(tiling.M) * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), static_cast<uint64_t>(tiling.Kb) * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), static_cast<uint64_t>(tiling.M) * tiling.N);

    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulUpperTriKernel<aType, bType, cType>::CalcTileInfo(
    int32_t blockIdx, const TCubeTiling &t, TileInfo &ti)
{
    const uint32_t mBlocks = CeilDivU32(static_cast<uint32_t>(t.M), static_cast<uint32_t>(t.singleCoreM));
    const int32_t mCoreIdx = static_cast<int32_t>(static_cast<uint32_t>(blockIdx) % mBlocks);
    const int32_t nCoreIdx = static_cast<int32_t>(static_cast<uint32_t>(blockIdx) / mBlocks);

    ti.rowStart = mCoreIdx * t.singleCoreM;
    ti.colStart = nCoreIdx * t.singleCoreN;

    ti.valid = (ti.rowStart < t.M) && (ti.colStart < t.N);
    if (!ti.valid) {
        ti.tileM = 0; ti.tileN = 0;
        ti.offsetA = 0; ti.offsetB = 0; ti.offsetC = 0;
        return;
    }

    ti.tileM = t.singleCoreM;
    ti.tileN = t.singleCoreN;
    if (ti.rowStart + ti.tileM > t.M) ti.tileM = t.M - ti.rowStart;
    if (ti.colStart + ti.tileN > t.N) ti.tileN = t.N - ti.colStart;

    ti.offsetA = ti.rowStart * t.Ka;
    ti.offsetB = ti.colStart;
    ti.offsetC = ti.rowStart * t.N + ti.colStart;
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulUpperTriKernel<aType, bType, cType>::EnsureZeroSlab(AscendC::TPipe *pipe)
{
    if (zeroReady_) return;
    pipe->InitBuffer(zeroBuf_, static_cast<uint32_t>(ZERO_SLAB_ELEMS * sizeof(cType)));
    zeroSlab_ = zeroBuf_.Get<cType>(ZERO_SLAB_ELEMS);
    AscendC::Duplicate(zeroSlab_, (cType)0, ZERO_SLAB_ELEMS);
    AscendC::PipeBarrier<PIPE_V>();
    zeroReady_ = true;
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulUpperTriKernel<aType, bType, cType>::ZeroFullTileLocal(
    AscendC::GlobalTensor<cType> cTile, int32_t Nstride, int32_t tileM, int32_t tileN)
{
    for (int32_t r = 0; r < tileM; ++r) {
        int32_t remaining = tileN;
        int32_t cOff = 0;
        while (remaining > 0) {
            const int32_t step = (remaining > ZERO_SLAB_ELEMS) ? ZERO_SLAB_ELEMS : remaining;
            const uint32_t gmOff = static_cast<uint32_t>(r * Nstride + cOff);
            AscendC::DataCopy(cTile[gmOff], zeroSlab_, static_cast<uint32_t>(step));
            remaining -= step;
            cOff += step;
        }
    }
    AscendC::PipeBarrier<PIPE_MTE3>();
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulUpperTriKernel<aType, bType, cType>::ZeroStrictLowerPrefixLocal(
    AscendC::GlobalTensor<cType> cTile, int32_t Nstride,
    int32_t rowStart, int32_t colStart, int32_t tileM, int32_t tileN)
{
    // For each row, zero columns where global_col < global_row (prefix inside this tile).
    for (int32_t i = 0; i < tileM; ++i) {
        const int32_t gr = rowStart + i;
        int32_t lowerCount = gr - colStart;
        if (lowerCount <= 0) continue;
        if (lowerCount > tileN) lowerCount = tileN;

        int32_t remaining = lowerCount;
        int32_t cOff = 0;
        while (remaining > 0) {
            const int32_t step = (remaining > ZERO_SLAB_ELEMS) ? ZERO_SLAB_ELEMS : remaining;
            const uint32_t gmOff = static_cast<uint32_t>(i * Nstride + cOff);
            AscendC::DataCopy(cTile[gmOff], zeroSlab_, static_cast<uint32_t>(step));
            remaining -= step;
            cOff += step;
        }
    }
    AscendC::PipeBarrier<PIPE_MTE3>();
}

template <typename aType, typename bType, typename cType>
template <bool setTmpSpace>
__aicore__ inline void MatmulUpperTriKernel<aType, bType, cType>::Process(AscendC::TPipe *pipe)
{
    if constexpr (setTmpSpace) {
        AscendC::TBuf<> tmpMMFormatUb;
        pipe->InitBuffer(tmpMMFormatUb, AscendC::TOTAL_VEC_LOCAL_SIZE);
        AscendC::LocalTensor<uint8_t> mmformatUb = tmpMMFormatUb.Get<uint8_t>(AscendC::TOTAL_VEC_LOCAL_SIZE);
        matmulObj.SetLocalWorkspace(mmformatUb);
    }

    TileInfo ti;
    CalcTileInfo(static_cast<int32_t>(GetBlockIdx()), tiling_, ti);
    if (!ti.valid) return;

    AscendC::GlobalTensor<aType> aTile = aGlobal[static_cast<uint32_t>(ti.offsetA)];
    AscendC::GlobalTensor<bType> bTile = bGlobal[static_cast<uint32_t>(ti.offsetB)];
    AscendC::GlobalTensor<cType> cTile = cGlobal[static_cast<uint32_t>(ti.offsetC)];

    // Fast path: if tile fully in strict-lower region, output is all zeros after triu => skip matmul.
    // Condition: max col in tile < min row in tile  => colStart + tileN <= rowStart
    if (ti.colStart + ti.tileN <= ti.rowStart) {
        EnsureZeroSlab(pipe);
        ZeroFullTileLocal(cTile, tiling_.N, ti.tileM, ti.tileN);
        return;
    }

    // Otherwise compute dense tile, then mask strict-lower part if needed.
    matmulObj.SetTensorA(aTile, /*isTransposeA=*/false);
    matmulObj.SetTensorB(bTile, /*isTransposeB=*/false);
    matmulObj.IterateAll(cTile);
    matmulObj.End();

    // If tile entirely on/above diagonal, nothing to mask.
    if (ti.colStart >= ti.rowStart + ti.tileM) {
        return;
    }

    EnsureZeroSlab(pipe);
    ZeroStrictLowerPrefixLocal(cTile, tiling_.N, ti.rowStart, ti.colStart, ti.tileM, ti.tileN);
}

extern "C" __global__ __aicore__ void matmul_for_upper_triangular_matrices_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    MatmulUpperTriKernel<float, float, float> kernel;
    AscendC::TPipe pipe;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), kernel.matmulObj, &tilingData.cubeTilingData);
    kernel.Init(a, b, c, workspace, tilingData.cubeTilingData);

    if (TILING_KEY_IS(1)) {
        kernel.Process(&pipe);
    } else if (TILING_KEY_IS(2)) {
        kernel.Process<true>(&pipe);
    }
}
