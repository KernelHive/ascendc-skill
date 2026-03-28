
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

template <typename aType, typename bType, typename cType>
class SquareMatmulKernel {
public:
    __aicore__ inline SquareMatmulKernel() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling);

    template <bool setTmpSpace = false>
    __aicore__ inline void Process(AscendC::TPipe *pipe);

    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                      int32_t &offsetA, int32_t &offsetB, int32_t &offsetC);

    __aicore__ inline uint32_t GetTotalTiles(const TCubeTiling &tiling);

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
__aicore__ inline void SquareMatmulKernel<aType, bType, cType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling)
{
    (void)workspace;
    this->tiling = tiling;

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
}

template <typename aType, typename bType, typename cType>
__aicore__ inline uint32_t SquareMatmulKernel<aType, bType, cType>::GetTotalTiles(const TCubeTiling &tiling)
{
    const uint32_t mTiles = Ceiling(static_cast<uint32_t>(tiling.M), static_cast<uint32_t>(tiling.singleCoreM));
    const uint32_t nTiles = Ceiling(static_cast<uint32_t>(tiling.N), static_cast<uint32_t>(tiling.singleCoreN));
    return mTiles * nTiles;
}

template <typename aType, typename bType, typename cType>
template <bool setTmpSpace>
__aicore__ inline void SquareMatmulKernel<aType, bType, cType>::Process(AscendC::TPipe *pipe)
{
    if constexpr (setTmpSpace) {
        AscendC::TBuf<> tmpMMFormatUb;
        AscendC::LocalTensor<uint8_t> mmformatUb;
        pipe->InitBuffer(tmpMMFormatUb, AscendC::TOTAL_VEC_LOCAL_SIZE);
        mmformatUb = tmpMMFormatUb.Get<uint8_t>(AscendC::TOTAL_VEC_LOCAL_SIZE);
        matmulObj.SetLocalWorkspace(mmformatUb);
    }

    // Stride over all tiles with multiple blocks to improve cube occupancy safely.
    const uint32_t totalTiles = GetTotalTiles(tiling);
    const uint32_t blockDim = static_cast<uint32_t>(GetBlockNum());
    const uint32_t start = static_cast<uint32_t>(GetBlockIdx());

    for (uint32_t tileId = start; tileId < totalTiles; tileId += blockDim) {
        int32_t offsetA = 0;
        int32_t offsetB = 0;
        int32_t offsetC = 0;
        CalcOffset(static_cast<int32_t>(tileId), tiling, offsetA, offsetB, offsetC);

        auto aTile = aGlobal[offsetA];
        auto bTile = bGlobal[offsetB];
        auto cTile = cGlobal[offsetC];

        matmulObj.SetTensorA(aTile);
        matmulObj.SetTensorB(bTile);
        matmulObj.IterateAll(cTile);
        matmulObj.End();
    }
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void SquareMatmulKernel<aType, bType, cType>::CalcOffset(
    int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB, int32_t &offsetC)
{
    // Tile over (M, N); K is handled internally by Matmul iteration/tiling.
    auto mSingleBlocks = Ceiling(static_cast<uint32_t>(tiling.M), static_cast<uint32_t>(tiling.singleCoreM));
    auto mCoreIndx = blockIdx % static_cast<int32_t>(mSingleBlocks);
    auto nCoreIndx = blockIdx / static_cast<int32_t>(mSingleBlocks);

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
}

extern "C" __global__ __aicore__ void square_matrix_multiplication_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    SquareMatmulKernel<float, float, float> matmulKernel;
    AscendC::TPipe pipe;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &tilingData.cubeTilingData);
    matmulKernel.Init(a, b, c, workspace, tilingData.cubeTilingData);

    if (TILING_KEY_IS(1)) {
        matmulKernel.Process<false>(&pipe);
    } else if (TILING_KEY_IS(2)) {
        matmulKernel.Process<true>(&pipe);
    }
}
