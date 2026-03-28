
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1U) / b;
}

template <typename aType, typename bType, typename cType>
class GemmKernel {
public:
    __aicore__ inline GemmKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight_t, GM_ADDR y,
                               GM_ADDR workspace, const TCubeTiling &tiling);
    template <bool setTmpSpace = false>
    __aicore__ inline void Process(AscendC::TPipe *pipe);

    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                      int32_t &offsetA, int32_t &offsetB, int32_t &offsetC);

    Matmul<
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>
    > matmulObj;

private:
    AscendC::GlobalTensor<aType> xGlobal;
    AscendC::GlobalTensor<bType> wTGlobal;
    AscendC::GlobalTensor<cType> yGlobal;
    TCubeTiling tiling;
};

template <typename aType, typename bType, typename cType>
__aicore__ inline void GemmKernel<aType, bType, cType>::Init(
    GM_ADDR x, GM_ADDR weight_t, GM_ADDR y, GM_ADDR workspace, const TCubeTiling &tiling)
{
    (void)workspace;
    this->tiling = tiling;

    xGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(x), tiling.M * tiling.Ka);
    wTGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(weight_t), tiling.Kb * tiling.N);
    yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(y), tiling.M * tiling.N);

    int32_t offsetA = 0, offsetB = 0, offsetC = 0;
    CalcOffset(static_cast<int32_t>(GetBlockIdx()), tiling, offsetA, offsetB, offsetC);
    xGlobal = xGlobal[offsetA];
    wTGlobal = wTGlobal[offsetB];
    yGlobal = yGlobal[offsetC];

    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
}

template <typename aType, typename bType, typename cType>
template <bool setTmpSpace>
__aicore__ inline void GemmKernel<aType, bType, cType>::Process(AscendC::TPipe *pipe)
{
    if constexpr (setTmpSpace) {
        AscendC::TBuf<> tmpMMFormatUb;
        AscendC::LocalTensor<uint8_t> mmformatUb;
        pipe->InitBuffer(tmpMMFormatUb, AscendC::TOTAL_VEC_LOCAL_SIZE);
        mmformatUb = tmpMMFormatUb.Get<uint8_t>(AscendC::TOTAL_VEC_LOCAL_SIZE);
        matmulObj.SetLocalWorkspace(mmformatUb);
    }

    matmulObj.SetTensorA(xGlobal);
    matmulObj.SetTensorB(wTGlobal);
    matmulObj.IterateAll(yGlobal);
    matmulObj.End();
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void GemmKernel<aType, bType, cType>::CalcOffset(
    int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB, int32_t &offsetC)
{
    auto mSingleBlocks = Ceiling(static_cast<uint32_t>(tiling.M), static_cast<uint32_t>(tiling.singleCoreM));
    auto mCoreIndx = static_cast<uint32_t>(blockIdx) % mSingleBlocks;
    auto nCoreIndx = static_cast<uint32_t>(blockIdx) / mSingleBlocks;

    offsetA = static_cast<int32_t>(mCoreIndx * static_cast<uint32_t>(tiling.Ka) * static_cast<uint32_t>(tiling.singleCoreM));
    offsetB = static_cast<int32_t>(nCoreIndx * static_cast<uint32_t>(tiling.singleCoreN));
    offsetC = static_cast<int32_t>(mCoreIndx * static_cast<uint32_t>(tiling.N) * static_cast<uint32_t>(tiling.singleCoreM)
                                 + nCoreIndx * static_cast<uint32_t>(tiling.singleCoreN));
}

extern "C" __global__ __aicore__ void gemm_custom(
    GM_ADDR x, GM_ADDR weight_t, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GemmKernel<bfloat16_t, bfloat16_t, bfloat16_t> op;
    AscendC::TPipe pipe;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj, &tilingData.cubeTilingData);
    op.Init(x, weight_t, y, workspace, tilingData.cubeTilingData);

    if (TILING_KEY_IS(1)) {
        op.Process(&pipe);
    } else if (TILING_KEY_IS(2)) {
        op.Process<true>(&pipe);
    }
}
