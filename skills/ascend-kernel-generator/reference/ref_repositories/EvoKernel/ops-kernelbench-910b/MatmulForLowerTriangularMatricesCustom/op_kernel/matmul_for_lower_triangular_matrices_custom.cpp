
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t CeilingU32(uint32_t a, uint32_t b)
{
    return (a + b - 1U) / b;
}

template <typename T>
__aicore__ inline T MinT(T a, T b) { return a < b ? a : b; }

template <typename aType, typename bType, typename cType>
class MatmulLowerTriKernel {
public:
    __aicore__ inline MatmulLowerTriKernel() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling);

    template <bool setTmpSpace = false>
    __aicore__ inline void Process(AscendC::TPipe *pipe);

private:
    __aicore__ inline void CalcTileInfo(int32_t blockIdx, const TCubeTiling &tiling,
                                        int32_t &offsetA, int32_t &offsetB, int32_t &offsetC,
                                        int32_t &rowStart, int32_t &colStart,
                                        int32_t &tileM, int32_t &tileN);

    __aicore__ inline void MaskBTileLowerTri(AscendC::TPipe *pipe,
                                            int32_t rowStart, int32_t colStart,
                                            int32_t tileM, int32_t tileN);

public:
    Matmul<
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>
    > matmulObj;

    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;

    // Per-block masked B tile in GM is avoided; we build a masked B tile in UB and run matmul against it.
    AscendC::TBuf<> bUbBuf_;
    AscendC::TBuf<> maskUbBuf_;

    TCubeTiling tiling_{};
};

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulLowerTriKernel<aType, bType, cType>::CalcTileInfo(
    int32_t blockIdx, const TCubeTiling &tiling,
    int32_t &offsetA, int32_t &offsetB, int32_t &offsetC,
    int32_t &rowStart, int32_t &colStart, int32_t &tileM, int32_t &tileN)
{
    const uint32_t mSingleBlocks = CeilingU32(static_cast<uint32_t>(tiling.M),
                                             static_cast<uint32_t>(tiling.singleCoreM));
    const int32_t mCoreIdx = static_cast<int32_t>(static_cast<uint32_t>(blockIdx) % mSingleBlocks);
    const int32_t nCoreIdx = static_cast<int32_t>(static_cast<uint32_t>(blockIdx) / mSingleBlocks);

    rowStart = mCoreIdx * tiling.singleCoreM;
    colStart = nCoreIdx * tiling.singleCoreN;

    tileM = tiling.singleCoreM;
    tileN = tiling.singleCoreN;
    if (rowStart + tileM > tiling.M) tileM = tiling.M - rowStart;
    if (colStart + tileN > tiling.N) tileN = tiling.N - colStart;

    offsetA = mCoreIdx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIdx * tiling.singleCoreN;
    offsetC = mCoreIdx * tiling.N * tiling.singleCoreM + nCoreIdx * tiling.singleCoreN;
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulLowerTriKernel<aType, bType, cType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling)
{
    (void)workspace;
    tiling_ = tiling;

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);

    int32_t offsetA = 0, offsetB = 0, offsetC = 0;
    int32_t rowStart = 0, colStart = 0, tileM = 0, tileN = 0;
    CalcTileInfo(static_cast<int32_t>(GetBlockIdx()), tiling, offsetA, offsetB, offsetC, rowStart, colStart, tileM, tileN);

    aGlobal = aGlobal[static_cast<uint32_t>(offsetA)];
    bGlobal = bGlobal[static_cast<uint32_t>(offsetB)];
    cGlobal = cGlobal[static_cast<uint32_t>(offsetC)];
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulLowerTriKernel<aType, bType, cType>::MaskBTileLowerTri(
    AscendC::TPipe *pipe, int32_t rowStart, int32_t colStart, int32_t tileM, int32_t tileN)
{
    // We want C(i,j)=sum_k A(i,k)*B(k,j) but only keep j<=i.
    // Equivalent: C = A @ (B * L), where L(k,j) = 1 if (globalCol=j) <= (globalRow=i) ??? (depends on i, not k).
    // A right-multiplicative mask depends on output row i, so it cannot be expressed solely as a mask on B.
    // Therefore we instead mask the output tile during accumulation by multiplying each output row i by a 0/1 mask
    // after matmul, but still in UB to avoid GM loops.
    // In this toolchain, we approximate by generating a per-row mask and applying it to the output tile in-place
    // using vector ops before GM writeback through matmul API (IterateAll writes to GM), so we must post-mask.
    // To keep this function meaningful and low-overhead, we only generate a small mask buffer and rely on a fast
    // vector write-zero on GM for the strict upper region using a single contiguous memset per row segment in UB,
    // then store back. However matmul API writes directly to GM; thus this method is disabled (kept for future).
    (void)pipe; (void)rowStart; (void)colStart; (void)tileM; (void)tileN;
}

template <typename aType, typename bType, typename cType>
template <bool setTmpSpace>
__aicore__ inline void MatmulLowerTriKernel<aType, bType, cType>::Process(AscendC::TPipe *pipe)
{
    if constexpr (setTmpSpace) {
        AscendC::TBuf<> tmpMMFormatUb;
        AscendC::LocalTensor<uint8_t> mmformatUb;
        pipe->InitBuffer(tmpMMFormatUb, AscendC::TOTAL_VEC_LOCAL_SIZE);
        mmformatUb = tmpMMFormatUb.Get<uint8_t>(AscendC::TOTAL_VEC_LOCAL_SIZE);
        matmulObj.SetLocalWorkspace(mmformatUb);
    }

    // Main matmul
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.IterateAll(cGlobal);
    matmulObj.End();

    // Replace expensive scalar-heavy postprocessing loop with a fast GM-side triangular clear:
    // do a vectorized "upper triangle clear" using large contiguous chunks per row, but without GM reads.
    int32_t offsetA = 0, offsetB = 0, offsetC = 0;
    int32_t rowStart = 0, colStart = 0, tileM = 0, tileN = 0;
    CalcTileInfo(static_cast<int32_t>(GetBlockIdx()), tiling_, offsetA, offsetB, offsetC, rowStart, colStart, tileM, tileN);

    if (colStart + tileN - 1 <= rowStart) {
        return;
    }

    constexpr int32_t CHUNK_ELEMS = 8192;
    AscendC::TBuf<> ubBuf;
    pipe->InitBuffer(ubBuf, static_cast<uint32_t>(CHUNK_ELEMS * sizeof(cType)));
    AscendC::LocalTensor<cType> ub = ubBuf.Get<cType>(CHUNK_ELEMS);

    for (int32_t i = 0; i < tileM; ++i) {
        const int32_t globalRow = rowStart + i;
        int32_t firstUpper = globalRow + 1;
        if (firstUpper < colStart) firstUpper = colStart;
        const int32_t localStart = firstUpper - colStart;
        if (localStart >= tileN) continue;

        int32_t remaining = tileN - localStart;
        int32_t colOff = localStart;

        while (remaining > 0) {
            const int32_t step = (remaining > CHUNK_ELEMS) ? CHUNK_ELEMS : remaining;
            AscendC::Duplicate(ub, (cType)0, step);
            AscendC::PipeBarrier<PIPE_V>();
            const uint32_t gmOff = static_cast<uint32_t>(i * tiling_.N + colOff);
            AscendC::DataCopy(cGlobal[gmOff], ub, static_cast<uint32_t>(step));
            remaining -= step;
            colOff += step;
        }
    }
}

extern "C" __global__ __aicore__ void matmul_for_lower_triangular_matrices_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    MatmulLowerTriKernel<float, float, float> kernel;
    AscendC::TPipe pipe;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), kernel.matmulObj, &tilingData.cubeTilingData);
    kernel.Init(a, b, c, workspace, tilingData.cubeTilingData);

    if (TILING_KEY_IS(1)) {
        kernel.Process(&pipe);
    } else if (TILING_KEY_IS(2)) {
        kernel.Process<true>(&pipe);
    }
}
