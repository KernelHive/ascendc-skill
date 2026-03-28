
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1U) / b;
}

template <typename aType, typename bType, typename cType>
class MatmulKernelLargeK {
public:
    __aicore__ inline MatmulKernelLargeK() {}
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

    AscendC::GlobalTensor<aType> aGlobal; // (M,K)
    AscendC::GlobalTensor<bType> bGlobal; // (K,N)
    AscendC::GlobalTensor<cType> cGlobal; // (M,N)
    TCubeTiling tiling;
};

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernelLargeK<aType, bType, cType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling)
{
    (void)workspace;
    this->tiling = tiling;

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
__aicore__ inline void MatmulKernelLargeK<aType, bType, cType>::Process(AscendC::TPipe *pipe)
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
__aicore__ inline void MatmulKernelLargeK<aType, bType, cType>::CalcOffset(
    int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB, int32_t &offsetC)
{
    // 2D core mapping over output tiles (M,N)
    auto mSingleBlocks = Ceiling(static_cast<uint32_t>(tiling.M), static_cast<uint32_t>(tiling.singleCoreM));
    auto mCoreIndx = blockIdx % static_cast<int32_t>(mSingleBlocks);
    auto nCoreIndx = blockIdx / static_cast<int32_t>(mSingleBlocks);

    const int32_t m0 = mCoreIndx * tiling.singleCoreM;
    const int32_t n0 = nCoreIndx * tiling.singleCoreN;

    // ND row-major:
    // A tile starts at (m0, 0) => offsetA = m0 * K
    // B tile starts at (0, n0) => offsetB = n0
    // C tile starts at (m0, n0) => offsetC = m0 * N + n0
    offsetA = m0 * tiling.Ka;
    offsetB = n0;
    offsetC = m0 * tiling.N + n0;
}

extern "C" __global__ __aicore__ void matmul_with_large_k_dimension_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    MatmulKernelLargeK<float, float, float> matmulKernel;
    AscendC::TPipe pipe;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &tilingData.cubeTilingData);
    matmulKernel.Init(a, b, c, workspace, tilingData.cubeTilingData);

    if (TILING_KEY_IS(1)) {
        matmulKernel.Process(&pipe);
    } else if (TILING_KEY_IS(2)) {
        matmulKernel.Process<true>(&pipe);
    }
}
