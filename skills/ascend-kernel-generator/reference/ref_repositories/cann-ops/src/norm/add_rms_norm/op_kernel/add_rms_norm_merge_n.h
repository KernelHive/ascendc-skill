/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_merge_n.h
 * \brief
 */
#ifndef ADD_RMS_NORM_MERGE_N_H_
#define ADD_RMS_NORM_MERGE_N_H_
#include "rms_norm_base.h"

using namespace AscendC;

template <typename T>
class KernelAddRmsNormMergeN {
public:
    __aicore__ inline KernelAddRmsNormMergeN(TPipe *pipe)
    {
        Ppipe = pipe;
    }
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR x, const AddRMSNormTilingData *tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        this->numRow = tiling->num_row;
        this->numCol = tiling->num_col;
        uint32_t numPerBlock = ONE_BLK_SIZE / sizeof(T);
        this->numColAlign = AlignUp(numCol, numPerBlock);
        this->blockFactor = tiling->block_factor;
        this->rowFactor = tiling->row_factor;
        this->ubFactor = tiling->ub_factor;
        this->epsilon = tiling->epsilon;
        this->avgFactor = (numCol != 0) ? (float)1.0 / numCol : 0;

        blockIdx_ = GetBlockIdx();
        if (blockIdx_ < GetBlockNum() - 1) {
            this->rowWork = blockFactor;
        } else if (blockIdx_ == GetBlockNum() - 1) {
            this->rowWork = numRow - (GetBlockNum() - 1) * blockFactor;
        } else {
        }
        // get start index for current core, core parallel
        x1Gm.SetGlobalBuffer((__gm__ T *)x1 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ T *)gamma, numCol);
        yGm.SetGlobalBuffer((__gm__ T *)y + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        rstdGm.SetGlobalBuffer((__gm__ float *)rstd + blockIdx_ * blockFactor, blockFactor);
        xGm.SetGlobalBuffer((__gm__ T *)x + blockIdx_ * blockFactor * numCol, rowWork * numCol);

        // pipe alloc memory to queue, the unit is Bytes
        Ppipe->InitBuffer(inQueueX, DOUBLE_BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(outQueueY, DOUBLE_BUFFER_NUM, ubFactor * sizeof(T));
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        Ppipe->InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * sizeof(float));
#else
        Ppipe->InitBuffer(rstdBuf, rowFactor * sizeof(float));
#endif
        if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
            Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        }
        Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(tmpBuf, rowFactor * NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
        BroadCastGamma(gammaLocal);
        uint32_t i_o_max = CeilDiv(rowWork, rowFactor);
        uint32_t row_tail = rowWork - (i_o_max - 1) * rowFactor;

        for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
            SubProcess(i_o, rowFactor, gammaLocal);
        }
        SubProcess(i_o_max - 1, row_tail, gammaLocal);
        inQueueGamma.FreeTensor(gammaLocal);
    }

    __aicore__ inline void SubProcess(uint32_t i_o, uint32_t calc_row_num, LocalTensor<T> &gammaLocal)
    {
        uint32_t gm_bias = i_o * rowFactor * numCol;
        uint32_t elementNum = calc_row_num * numColAlign;
        CopyInX(gm_bias, calc_row_num);
        LocalTensor<T> xLocal = ComputeX(elementNum);
        CopyOutX(gm_bias, calc_row_num);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        ComputeRstd(xLocal, rstdLocal, calc_row_num);
        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(i_o, calc_row_num);
#else
        LocalTensor<float> rstdLocal = rstdBuf.Get<float>();
        ComputeRstd(xLocal, rstdLocal, calc_row_num);
#endif
        ComputeY(xLocal, gammaLocal, rstdLocal, calc_row_num);
        CopyOutY(gm_bias, calc_row_num);
    }

private:
    __aicore__ inline void CopyInX(uint32_t gm_bias, uint32_t calc_row_num)
    {
        LocalTensor<T> x1Local = inQueueX.AllocTensor<T>();
        if (isNumColAlign) {
            DataCopyCustom<T>(x1Local, x1Gm[gm_bias], calc_row_num * numCol);
        } else {
            DataCopyCustom<T>(x1Local, x1Gm[gm_bias], calc_row_num, numCol);
        }
        inQueueX.EnQue(x1Local);
        LocalTensor<T> x2Local = inQueueX.AllocTensor<T>();
        if (isNumColAlign) {
            DataCopyCustom<T>(x2Local, x2Gm[gm_bias], calc_row_num * numCol);
        } else {
            DataCopyCustom<T>(x2Local, x2Gm[gm_bias], calc_row_num, numCol);
        }
        inQueueX.EnQue(x2Local);
    }

    __aicore__ inline LocalTensor<T> ComputeX(uint32_t elementNum)
    {
        LocalTensor<T> x1Local = inQueueX.DeQue<T>();
        LocalTensor<T> x2Local = inQueueX.DeQue<T>();
        LocalTensor<T> xLocal = outQueueY.AllocTensor<T>();
        if constexpr (!IsSame<T, bfloat16_t>::value) {
            Add(xLocal, x1Local, x2Local, elementNum);
        } else {
            LocalTensor<float> x1Fp32 = xFp32Buf.Get<float>();
            LocalTensor<float> x2Fp32 = sqxBuf.Get<float>();
            Cast(x1Fp32, x1Local, RoundMode::CAST_NONE, elementNum);
            Cast(x2Fp32, x2Local, RoundMode::CAST_NONE, elementNum);
            PipeBarrier<PIPE_V>();
            Add(x1Fp32, x1Fp32, x2Fp32, elementNum);
            PipeBarrier<PIPE_V>();
            Cast(xLocal, x1Fp32, RoundMode::CAST_RINT, elementNum);
        }
        inQueueX.FreeTensor(x1Local);
        inQueueX.FreeTensor(x2Local);
        outQueueY.EnQue(xLocal);
        PipeBarrier<PIPE_V>();
        return xLocal;
    }

    __aicore__ inline void CopyOutX(uint32_t gm_bias, uint32_t calc_row_num)
    {
        // CopyOut x1 + x2
        auto xOut = outQueueY.DeQue<T>();
        if (isNumColAlign) {
            DataCopyCustom<T>(xGm[gm_bias], xOut, calc_row_num * numCol);
        } else {
            DataCopyCustom<T>(xGm[gm_bias], xOut, calc_row_num, numCol);
        }
        outQueueY.FreeTensor(xOut);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void BroadCastGamma(LocalTensor<T> &gammaLocal)
    {
        const uint32_t srcShape[2] = {1, numColAlign};
        const uint32_t dstShape[2] = {rowFactor, numColAlign};
        LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();
        if constexpr (IsSame<T, bfloat16_t>::value) {
            LocalTensor<half> interpreLocal = gammaLocal.template ReinterpretCast<half>();
            BroadCast<half, DIM_NUM, 0>(interpreLocal, interpreLocal, dstShape, srcShape, tmpLocal);
        } else {
            BroadCast<T, DIM_NUM, 0>(gammaLocal, gammaLocal, dstShape, srcShape, tmpLocal);
        }
    }

    __aicore__ inline void ComputeRstd(LocalTensor<T> xLocal, LocalTensor<float> rstdLocal, uint32_t calc_row_num)
    {
        uint32_t elementNum = calc_row_num * numColAlign;
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        if constexpr (!IsSame<T, float>::value) {
            LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
            Cast(x_fp32, xLocal, RoundMode::CAST_NONE, elementNum);
            PipeBarrier<PIPE_V>();
            Mul(sqx, x_fp32, x_fp32, elementNum);
        } else {
            Mul(sqx, xLocal, xLocal, elementNum);
        }
        PipeBarrier<PIPE_V>();

        Muls(sqx, sqx, avgFactor, elementNum);
        PipeBarrier<PIPE_V>();

        ReduceSumMultiN(rstdLocal, sqx, tmpLocal, calc_row_num, numCol, numColAlign);
        PipeBarrier<PIPE_V>();
        Adds(rstdLocal, rstdLocal, epsilon, calc_row_num);
        PipeBarrier<PIPE_V>();

        Sqrt(rstdLocal, rstdLocal, calc_row_num);
        Duplicate(tmpLocal, ONE, calc_row_num);
        PipeBarrier<PIPE_V>();

        Div(rstdLocal, tmpLocal, rstdLocal, calc_row_num);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeY(
        LocalTensor<T> xLocal, LocalTensor<T> gammaLocal, LocalTensor<float> rstdLocal, uint32_t calc_row_num)
    {
        uint32_t elementNum = calc_row_num * numColAlign;
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        auto sharedTmpLocal = tmpBuf.Get<uint8_t>();
        const uint32_t srcShape[2] = {calc_row_num, 1};
        const uint32_t dstShape[2] = {calc_row_num, numColAlign};
        BroadCast<float, DIM_NUM, 1>(sqx, rstdLocal, dstShape, srcShape, sharedTmpLocal);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        if constexpr (!IsSame<T, float>::value) {
            LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
            Mul(x_fp32, x_fp32, sqx, elementNum);
            PipeBarrier<PIPE_V>();
            if constexpr (IsSame<T, half>::value) {
                Cast(yLocal, x_fp32, RoundMode::CAST_NONE, elementNum);
            } else {
                Cast(yLocal, x_fp32, RoundMode::CAST_RINT, elementNum);
            }
        } else {
            Mul(yLocal, xLocal, sqx, elementNum);
        }
        PipeBarrier<PIPE_V>();

        if constexpr (!IsSame<T, bfloat16_t>::value) {
            Mul(yLocal, yLocal, gammaLocal, elementNum);
        } else {
            LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
            Cast(x_fp32, yLocal, RoundMode::CAST_NONE, elementNum);
            Cast(sqx, gammaLocal, RoundMode::CAST_NONE, elementNum);
            PipeBarrier<PIPE_V>();
            Mul(x_fp32, x_fp32, sqx, elementNum);
            PipeBarrier<PIPE_V>();
            Cast(yLocal, x_fp32, RoundMode::CAST_RINT, elementNum);
        }
        PipeBarrier<PIPE_V>();
        outQueueY.EnQue<T>(yLocal);
    }

    __aicore__ inline void CopyOutY(uint32_t progress, uint32_t calc_row_num)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        if (isNumColAlign) {
            DataCopyCustom<T>(yGm[progress], yLocal, calc_row_num * numCol);
        } else {
            DataCopyCustom<T>(yGm[progress], yLocal, calc_row_num, numCol);
        }
        outQueueY.FreeTensor(yLocal);
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    __aicore__ inline void CopyOutRstd(uint32_t outer_progress, uint32_t num)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
        DataCopyCustom<float>(rstdGm[outer_progress * rowFactor], rstdLocal, num);
        outQueueRstd.FreeTensor(rstdLocal);
    }
#endif

private:
    TPipe *Ppipe = nullptr;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGamma;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX;
    // create queues for output, in this case depth is equal to buffer num
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueRstd;
#else
    TBuf<TPosition::VECCALC> rstdBuf;
#endif
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueY;

    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> sqxBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<T> xGm;

    uint32_t numRow;
    uint32_t numCol;
    uint32_t numColAlign;
    uint32_t blockFactor;  // number of calculations rows on each core
    uint32_t rowFactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    int32_t blockIdx_;
    uint32_t rowWork = 1;
    bool isNumColAlign;
};
#endif  // ADD_RMS_NORM_MERGE_N_H_