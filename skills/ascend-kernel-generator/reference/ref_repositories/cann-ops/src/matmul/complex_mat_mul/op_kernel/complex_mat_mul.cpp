/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file complex_mat_mul.cpp
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

template <typename T>
__aicore__ inline T Min(T x, T y) { return x < y ? x : y; }

template <typename T>
__aicore__ inline T Max(T x, T y) { return x > y ? x : y; }

using namespace AscendC;
using namespace matmul;

__aicore__ inline uint32_t AlignUP(uint32_t x, uint32_t p)
{
    return (x + (p - 1)) & ~(p - 1);
}

__aicore__ inline uint32_t AlignDown(uint32_t x, uint32_t p)
{
    return x & ~(p - 1);
}

template <uint32_t LEN, uint32_t TILE_M, uint32_t TILE_N, uint32_t Alignment = 64>
class KernelMatMulComplex
{
  public:
    __aicore__ inline KernelMatMulComplex()
    {
    }
    __aicore__ inline void Init(TPipe *pipe, GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                GM_ADDR bias, GM_ADDR workspace, uint32_t BatchSize, uint32_t M,
                                uint32_t K, uint32_t N, TCubeTiling *tiling)
    {
        this->BatchSize = BatchSize;
        this->M = M;
        this->K = K;
        this->N = N;
        this->M0 = AlignUP(M, TILE_M);
        this->K0 = AlignUP(K, Alignment);
        this->N0 = AlignUP(N, TILE_N);
        this->tiling = tiling;

        x1Gm.SetGlobalBuffer((__gm__ float *)x1, 2 * M * K * sizeof(float));
        x2Gm.SetGlobalBuffer((__gm__ float *)x2, 2 * K * N * sizeof(float));
        yGm.SetGlobalBuffer((__gm__ float *)y, 2 * M * N * sizeof(float));
        workGm.SetGlobalBuffer((__gm__ float *)workspace, (3 * M0 * K0 + 3 * K0 * N0) * sizeof(float));
        zGm.SetGlobalBuffer((__gm__ float *)workspace + (3 * M0 * K0 + 3 * K0 * N0), 3 * M0 * N0 * sizeof(float));

        if (bias)
        {
            biasGm.SetGlobalBuffer((__gm__ float *)bias, 2 * M * N * sizeof(float));
            pipe->InitBuffer(inQueueBias, 1, 2 * TILE_M * TILE_N * sizeof(float));
            addBias = true;
        }
        else
        {
            addBias = false;
        }

        pipe->InitBuffer(inQueueComplex, 1, 2 * LEN * sizeof(float));
        pipe->InitBuffer(outQueueWork, 1, 3 * LEN * sizeof(float));

        pipe->InitBuffer(outQueueY, 1, 2 * TILE_M * TILE_N * sizeof(float));
        pipe->InitBuffer(tBufAcc, (2 * TILE_M * TILE_N) * sizeof(float));
        pipe->InitBuffer(inQueueP, 2, (TILE_M * TILE_N) * sizeof(float));
        pipe->InitBuffer(tBufOffset,
                         (128) * sizeof(uint32_t) + 128 * sizeof(float));
        LocalTensor<uint32_t> offset = tBufOffset.Get<uint32_t>();

        for (int i = 0; i < 64; i++)
        {
            offset.SetValue(2 * i + 0, 4 * i);
            offset.SetValue(2 * i + 1, 4 * i + 256);
        }
    }

    __aicore__ inline void CalcOffset(uint32_t blockIdx, const TCubeTiling &tiling, uint32_t &offsetA, uint32_t &offsetB, uint32_t &offsetC)
    {
        uint32_t mSingleBlocks = (tiling.M - 1) / tiling.singleCoreM + 1;
        uint32_t mCoreIndx = blockIdx % mSingleBlocks;
        uint32_t nCoreIndx = blockIdx / mSingleBlocks;

        offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
        offsetB = nCoreIndx * tiling.singleCoreN;
        offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;

        if (mCoreIndx * tiling.singleCoreM >= tiling.M || nCoreIndx * tiling.singleCoreN >= tiling.N)
        {
            singleCoreM = 0;
            singleCoreN = 0;
            return;
        }

        uint32_t tailM = Min(tiling.M - mCoreIndx * tiling.singleCoreM, (uint32_t) tiling.singleCoreM);
        uint32_t tailN = Min(tiling.N - nCoreIndx * tiling.singleCoreN, (uint32_t) tiling.singleCoreN);
        if (tailM < tiling.singleCoreM || tailN < tiling.singleCoreN)
        {
            matmulObj.SetTail(tailM, tailN);
            singleCoreM = tailM;
            singleCoreN = tailN;
        }
        else
        {
            singleCoreM = tiling.singleCoreM;
            singleCoreN = tiling.singleCoreN;
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t offsetA, offsetB, offsetC;
        CalcOffset(GetBlockIdx(), *tiling, offsetA, offsetB, offsetC);

        const int cnt = GetBlockNum() * GetTaskRation();
        for (uint32_t b = 0; b < BatchSize; b++)
        {
            int idx = 0;

            for (uint32_t i = GetBlockIdx(); i < M; i += cnt)
            {
                for (uint32_t j = 0; j < K; j += LEN)
                {
                    const uint32_t len = Min(LEN, K - j);
                    CopyInX<true>(x1Gm[2 * (i * K + j)], 2 * len);
                    SplitAndAdd();
                    CopyOutWork(i * K0 + j, M0 * K0, AlignUP(len, Alignment));
                }
            }

            for (uint32_t i = GetBlockIdx(); i < K; i += cnt)
            {
                for (uint32_t j = 0; j < N; j += LEN)
                {
                    const uint32_t len = Min(LEN, N - j);
                    CopyInX(x2Gm[2 * (i * N + j)], 2 * len);
                    SplitAndAdd();
                    CopyOutWork(3 * M0 * K0 + i * N0 + j, K0 * N0, AlignUP(len, Alignment));
                }
            }

            SyncAll();

            BatchedGemm(zGm[offsetC], workGm[offsetA], workGm[offsetB + 3 * M0 * K0]);

            SyncAll();

            for (uint32_t i = 0; i < M; i += TILE_M, idx += N / TILE_N)
            {
                uint32_t len_m = Min(TILE_M, M - i);
                uint32_t j_start = TILE_N * ((GetBlockIdx() + cnt - idx % cnt) % cnt);
                for (uint32_t j = j_start; j < N; j += TILE_N * cnt)
                {
                    uint32_t len_n = Min(TILE_N, N - j);
                    if (addBias)
                    {
                        CopyInBias(i, j, len_m, len_n);
                    }
                    Compute(i, j);
                    CopyOut(i, j, len_m, len_n);
                }
            }
            x1Gm = x1Gm[2 * M * K];
            x2Gm = x2Gm[2 * K * N];
            yGm = yGm[2 * M * N];
            biasGm = biasGm[2 * M * N];
            SyncAll();
        }
    }

    template<bool need_reset = false>
    __aicore__ inline void CopyInX(const GlobalTensor<float> &ComplexGm, uint32_t len)
    {
        LocalTensor<float> xLocal = inQueueComplex.AllocTensor<float>();
        if (need_reset && AlignDown(len, 2 * Alignment) != len)
        {
            Duplicate(xLocal[AlignDown(len, 2 * Alignment)], 0.0f, 2 * Alignment);
        }
        DataCopyExtParams copyParamsX{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, ComplexGm, copyParamsX, padParams);
        inQueueComplex.EnQue(xLocal);
    }

    __aicore__ inline void SplitAndAdd()
    {
        LocalTensor<float> complex = inQueueComplex.DeQue<float>();
        LocalTensor<float> work = outQueueWork.AllocTensor<float>();

        Split(complex, work, work[LEN], LEN);
        Add(work[2 * LEN], work, work[LEN], LEN);

        inQueueComplex.FreeTensor(complex);
        outQueueWork.EnQue(work);
    }

    __aicore__ inline void CopyOutWork(uint32_t offset, uint32_t stride, uint32_t len)
    {
        LocalTensor<float> work = outQueueWork.DeQue<float>();
        DataCopyExtParams copyParamsX{3, static_cast<uint32_t>(len * sizeof(float)), (LEN - len) / 8, static_cast<uint32_t>((stride - len) * sizeof(float)), 0};
        DataCopyPad(workGm[offset], work, copyParamsX);
        outQueueWork.FreeTensor(work);
    }

    __aicore__ inline void CopyInBias(uint32_t i, uint32_t j, uint32_t rows,
                                      uint32_t cols)
    {
        LocalTensor<float> xLocal = inQueueBias.AllocTensor<float>();
        DataCopyExtParams copyParamsX;
        copyParamsX.blockCount = rows;
        copyParamsX.blockLen = 2 * cols * sizeof(float);
        copyParamsX.srcStride = 2 * (N - cols) * sizeof(float);
        copyParamsX.dstStride = (2 * (TILE_N - cols)) / (32 / sizeof(float));
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

        DataCopyPad(xLocal, biasGm[2 * (i * N + j)], copyParamsX, padParams);
        inQueueBias.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t i, uint32_t j, uint32_t rows,
                                   uint32_t cols)
    {
        LocalTensor<float> xLocal = outQueueY.DeQue<float>();
        DataCopyExtParams copyParamsX;
        copyParamsX.blockCount = rows;
        copyParamsX.blockLen = 2 * cols * sizeof(float);
        copyParamsX.srcStride = (2 * (TILE_N - cols)) / (32 / sizeof(float));
        copyParamsX.dstStride = (2 * (N - cols)) * sizeof(float);
        DataCopyPad(yGm[2 * (i * N + j)], xLocal, copyParamsX);
        outQueueY.FreeTensor(xLocal);
    }

    __aicore__ inline void BatchedGemm(const GlobalTensor<float> &z,
                                       const GlobalTensor<float> &x,
                                       const GlobalTensor<float> &y)
    {
        if (singleCoreM == 0 || singleCoreN == 0)
            return;

        matmulObj.SetTensorA(x);
        matmulObj.SetTensorB(y);
        matmulObj.IterateAll(z);
        matmulObj.SetTensorA(x[M0 * K0]);
        matmulObj.SetTensorB(y[K0 * N0]);
        matmulObj.IterateAll(z[M0 * N0]);
        matmulObj.SetTensorA(x[2 * M0 * K0]);
        matmulObj.SetTensorB(y[2 * K0 * N0]);
        matmulObj.IterateAll(z[2 * M0 * N0]);
    }

    __aicore__ inline void Split(const LocalTensor<float> &xComplex,
                                 const LocalTensor<float> &xReal,
                                 const LocalTensor<float> &xImag,
                                 uint32_t count)
    {
        LocalTensor<uint32_t> offset = tBufOffset.Get<uint32_t>();
        uint64_t rsvdCnt;
        GatherMask(xReal, xComplex, 1, false, 0, {1, static_cast<uint16_t>(count / 32), 8, 8}, rsvdCnt);
        GatherMask(xImag, xComplex, 2, false, 0, {1, static_cast<uint16_t>(count / 32), 8, 8}, rsvdCnt);
    }

    __aicore__ inline void Combine(const LocalTensor<float> &xComplex,
                                   const LocalTensor<float> &xReal,
                                   const LocalTensor<float> &xImag,
                                   uint32_t count)
    {
        LocalTensor<uint32_t> offset = tBufOffset.Get<uint32_t>();
        LocalTensor<float> tmp = tBufOffset.Get<float>()[128];

        for (int i = 0; i < count; i += 64)
        {
            DataCopy(tmp, xReal[i], 64);
            DataCopy(tmp[64], xImag[i], 64);
            Gather(xComplex[2 * i], tmp, offset, 0, 128);
        }
    }

    __aicore__ inline void CopyInP(uint32_t offset, uint32_t i, uint32_t j)
    {
        auto z = inQueueP.AllocTensor<float>();
        DataCopyParams params;
        params.blockCount = TILE_M;
        params.blockLen = TILE_N * sizeof(float) / 32;
        params.srcStride = (N0 - TILE_N) * sizeof(float) / 32;
        params.dstStride = 0;
        DataCopy(z, zGm[offset + i * N0 + j], params);
        inQueueP.EnQue(z);
    }

    __aicore__ inline void Compute(uint32_t i, uint32_t j)
    {
        LocalTensor<float> zReal = tBufAcc.Get<float>()[0];
        LocalTensor<float> zImag = tBufAcc.Get<float>()[TILE_M * TILE_N];

        /*
          z_real = x_real @ y_real - x_imag @ y_imag
          z_imag = x_real @ y_imag + x_imag @ y_real
        */

        CopyInP(0, i, j);
        auto p1 = inQueueP.DeQue<float>();
        DataCopy(zReal, p1, TILE_M * TILE_N);
        Muls(zImag, p1, -1.0f, TILE_M * TILE_N);
        inQueueP.FreeTensor(p1);

        CopyInP(M0 * N0, i, j);
        auto p2 = inQueueP.DeQue<float>();
        Sub(zReal, zReal, p2, TILE_M * TILE_N);
        Sub(zImag, zImag, p2, TILE_M * TILE_N);
        inQueueP.FreeTensor(p2);

        CopyInP(2 * M0 * N0, i, j);
        auto p3 = inQueueP.DeQue<float>();
        Add(zImag, zImag, p3, TILE_M * TILE_N);
        inQueueP.FreeTensor(p3);

        auto acc = outQueueY.AllocTensor<float>();
        Combine(acc, zReal, zImag, TILE_M * TILE_N);
        if (addBias)
        {
            auto bias = inQueueBias.DeQue<float>();
            Add(acc, acc, bias, 2 * TILE_M * TILE_N);
            inQueueBias.FreeTensor(bias);
        }
        outQueueY.EnQue(acc);
    }

    TQue<QuePosition::VECIN, 1> inQueueComplex, inQueueBias, inQueueP;
    TQue<QuePosition::VECOUT, 1> outQueueWork, outQueueY;

    GlobalTensor<float> x1Gm, x2Gm, yGm, zGm, biasGm, workGm;
    TBuf<TPosition::VECCALC> tBufOffset, tBufAcc;

    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>, CFG_MDL>
        matmulObj;

    TCubeTiling *tiling;
    uint32_t BatchSize, M, K, N, M0, K0, N0, singleCoreM, singleCoreN;
    bool addBias;
};

// (M, K) (K, N) (M, N)
extern "C" __global__ __aicore__ void complex_mat_mul(GM_ADDR x, GM_ADDR y,
                                              GM_ADDR bias, GM_ADDR z,
                                              GM_ADDR workspace,
                                              GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const uint32_t BatchSize = tilingData.BatchSize;
    const uint32_t M = tilingData.M;
    const uint32_t K = tilingData.K;
    const uint32_t N = tilingData.N;

    TPipe pipe;

    KernelMatMulComplex<2048, 64, 64> op;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj,
                      &tilingData.cubeTilingData);
    op.Init(&pipe, x, y, z, bias, workspace, BatchSize, M, K, N, &tilingData.cubeTilingData);
    op.Process();
}
