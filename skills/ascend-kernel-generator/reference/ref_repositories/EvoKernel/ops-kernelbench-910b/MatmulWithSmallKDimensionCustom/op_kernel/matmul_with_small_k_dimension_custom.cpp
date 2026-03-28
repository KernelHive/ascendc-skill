
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

template <typename aType, typename bType, typename cType>
class MatmulKernelSmallK {
public:
    __aicore__ inline MatmulKernelSmallK() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling);

    template <bool setTmpSpace = false>
    __aicore__ inline void Process(AscendC::TPipe *pipe);

    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                      int32_t &offsetA, int32_t &offsetB, int32_t &offsetC);

    Matmul<
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>
    > matmulObj;

    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;
    TCubeTiling tiling;
};

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernelSmallK<aType, bType, cType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling)
{
    (void)workspace;
    this->tiling = tiling;

    // A: (M,Ka), B: (Kb,N), C: (M,N)
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), static_cast<uint64_t>(tiling.M) * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), static_cast<uint64_t>(tiling.Kb) * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), static_cast<uint64_t>(tiling.M) * tiling.N);

    int32_t offsetA = 0;
    int32_t offsetB = 0;
    int32_t offsetC = 0;
    CalcOffset(static_cast<int32_t>(GetBlockIdx()), tiling, offsetA, offsetB, offsetC);

    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];

    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
}

template <typename aType, typename bType, typename cType>
template <bool setTmpSpace>
__aicore__ inline void MatmulKernelSmallK<aType, bType, cType>::Process(AscendC::TPipe *pipe)
{
    if constexpr (setTmpSpace) {
        AscendC::TBuf<> tmpMMFormatUb;
        AscendC::LocalTensor<uint8_t> mmformatUb;
        pipe->InitBuffer(tmpMMFormatUb, AscendC::TOTAL_VEC_LOCAL_SIZE);
        mmformatUb = tmpMMFormatUb.Get<uint8_t>(AscendC::TOTAL_VEC_LOCAL_SIZE);
        matmulObj.SetLocalWorkspace(mmformatUb);
    }

    matmulObj.SetTensorA(aGlobal, /*isTransposeA=*/false);
    matmulObj.SetTensorB(bGlobal, /*isTransposeB=*/false);
    matmulObj.IterateAll(cGlobal);
    matmulObj.End();
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernelSmallK<aType, bType, cType>::CalcOffset(
    int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB, int32_t &offsetC)
{
    // Tile over output (M,N) grid. Each block produces one (singleCoreM x singleCoreN) tile of C.
    const uint32_t mSingleBlocks = Ceiling(static_cast<uint32_t>(tiling.M), static_cast<uint32_t>(tiling.singleCoreM));
    const int32_t mCoreIndx = blockIdx % static_cast<int32_t>(mSingleBlocks);
    const int32_t nCoreIndx = blockIdx / static_cast<int32_t>(mSingleBlocks);

    const int32_t m0 = mCoreIndx * tiling.singleCoreM;
    const int32_t n0 = nCoreIndx * tiling.singleCoreN;

    // Row-major ND:
    // A base: row m0 => m0*Ka
    // B base: col tile starts at n0 => offset n0 (Matmul internal uses N strides)
    // But for ND B as (K,N), base element for column n0 is just +n0 (within first row);
    // Matmul API will handle correct 2D addressing given base pointer and tiling.
    offsetA = m0 * tiling.Ka;
    offsetB = n0;
    offsetC = m0 * tiling.N + n0;
}

extern "C" __global__ __aicore__ void matmul_with_small_k_dimension_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    MatmulKernelSmallK<float, float, float> op;
    AscendC::TPipe pipe;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj, &tilingData.cubeTilingData);
    op.Init(a, b, c, workspace, tilingData.cubeTilingData);

    if (TILING_KEY_IS(1)) {
        op.Process(&pipe);
    } else if (TILING_KEY_IS(2)) {
        op.Process<true>(&pipe);
    }
}
