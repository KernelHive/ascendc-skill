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
 * \file rms_norm_merge_n.h
 * \brief
 */
#ifndef RMS_NORM_MERGE_N_H_
#define RMS_NORM_MERGE_N_H_
#include "rms_norm_base.h"

using namespace AscendC;

template <typename T, typename T_GAMMA>
class KernelRmsNormMergeN : KernelRmsNormBase<T, T_GAMMA> {
public:
    __aicore__ inline KernelRmsNormMergeN()
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, const RMSNormTilingData *tiling)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        InitVar(tiling);
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ < GetBlockNum() - 1) {
            this->rowWork = blockFactor;
        } else if (blockIdx_ == GetBlockNum() - 1) {
            this->rowWork = numRow - (GetBlockNum() - 1) * blockFactor;
        }
        isNumColAlign = (numCol == numColAlign);
        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T *)x + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ T_GAMMA *)gamma, numCol);
        yGm.SetGlobalBuffer((__gm__ T *)y + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        rstdGm.SetGlobalBuffer((__gm__ float *)rstd + blockIdx_ * blockFactor, blockFactor);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, ubFactor * sizeof(T));
        pipe.InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(T_GAMMA));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, ubFactor * sizeof(T));
        pipe.InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * sizeof(float));

        if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
            pipe.InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        }
        pipe.InitBuffer(sqxBuf, ubFactor * sizeof(float));
        pipe.InitBuffer(tmpBuf, rowFactor * NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void InitVar(const RMSNormTilingData *tiling)
    {
        numRow = tiling->num_row;
        numCol = tiling->num_col;
        numColAlign = tiling->num_col_align;
        blockFactor = tiling->block_factor;
        rowFactor = tiling->row_factor;
        ubFactor = tiling->ub_factor;
        epsilon = tiling->epsilon;
        if (numCol != 0) {
            avgFactor = (float)1.0 / numCol;
        } else {
            avgFactor = 0;
        }
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<T_GAMMA> gammaLocal = inQueueGamma.DeQue<T_GAMMA>();
        BroadCastGamma(gammaLocal);
        uint32_t i_o_max = CeilDiv(rowWork, rowFactor);
        uint32_t row_tail = rowWork - (i_o_max - 1) * rowFactor;

        for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
            SubProcess(i_o, rowFactor, gammaLocal);
        }
        SubProcess(i_o_max - 1, row_tail, gammaLocal);
        inQueueGamma.FreeTensor(gammaLocal);
    }

    __aicore__ inline void SubProcess(uint32_t iOuter, uint32_t calcRowNum, LocalTensor<T_GAMMA> &gammaLocal)
    {
        uint32_t gmBias = iOuter * rowFactor * numCol;
        CopyIn(gmBias, calcRowNum);
        Compute(iOuter, gammaLocal, calcRowNum);
        CopyOutY(gmBias, calcRowNum);
    }

private:
    __aicore__ inline void CopyIn(uint32_t gm_bias, uint32_t calc_row_num)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        if (isNumColAlign) {
            DataCopyCustom<T>(xLocal, xGm[gm_bias], calc_row_num * numCol);
        } else {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            // only support v220
            DataCopyParams copyParams;
            copyParams.blockLen = numCol * sizeof(T);
            copyParams.blockCount = calc_row_num;
            DataCopyPadParams padParams;
            DataCopyPad(xLocal, xGm[gm_bias], copyParams, padParams);
#endif
        }

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T_GAMMA> gammaLocal = inQueueGamma.AllocTensor<T_GAMMA>();
        DataCopyCustom<T_GAMMA>(gammaLocal, gammaGm, numCol);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void BroadCastGamma(LocalTensor<T_GAMMA> &gammaLocal)
    {
        const uint32_t srcShape[2] = {1, numColAlign};
        const uint32_t dstShape[2] = {rowFactor, numColAlign};
        LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();
        if constexpr (IsSame<T_GAMMA, bfloat16_t>::value) {
            LocalTensor<half> interpreLocal = gammaLocal.template ReinterpretCast<half>();
            BroadCast<half, DIM_NUM, 0>(interpreLocal, interpreLocal, dstShape, srcShape, tmpLocal);
        } else {
            BroadCast<T_GAMMA, DIM_NUM, 0>(gammaLocal, gammaLocal, dstShape, srcShape, tmpLocal);
        }
    }

    __aicore__ inline void ComputeMulGammaCast(LocalTensor<T_GAMMA> gammaLocal, uint32_t elementNum)
    {
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();

        if constexpr (IsSame<T, float>::value) {
            Mul(yLocal, sqx, gammaLocal, elementNum);
        } else {
            if constexpr (IS_MIX_DTYPE) {
                Mul(sqx, sqx, gammaLocal, elementNum);
            } else {
                LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
                Cast(xFp32, gammaLocal, RoundMode::CAST_NONE, elementNum);
                AscendC::PipeBarrier<PIPE_V>();
                Mul(sqx, sqx, xFp32, elementNum);
            }
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (IsSame<T, half>::value) {
                Cast(yLocal, sqx, RoundMode::CAST_NONE, elementNum);
            } else {
                Cast(yLocal, sqx, RoundMode::CAST_RINT, elementNum);
            }
        }
        outQueueY.EnQue<T>(yLocal);
    }

    __aicore__ inline void Compute(uint32_t iOuter, LocalTensor<T_GAMMA> gammaLocal, uint32_t calcRowNum)
    {
        uint32_t elementNum = calcRowNum * numColAlign;
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        LocalTensor<float> xFp32;

        // 1. Cast x and Cal x^2
        if constexpr (IsSame<T, float>::value) {
            Mul(sqx, xLocal, xLocal, elementNum);
        } else {
            xFp32 = xFp32Buf.Get<float>();
            Cast(xFp32, xLocal, RoundMode::CAST_NONE, elementNum);
            inQueueX.FreeTensor(xLocal);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(sqx, xFp32, xFp32, elementNum);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // 2. Rstd = 1 / sqrt(1 / reduceDim * reducesum(x^2) + eps)
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        ReduceSumMultiN(rstdLocal, sqx, tmpLocal, calcRowNum, numCol, numColAlign);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(rstdLocal, rstdLocal, avgFactor, calcRowNum);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(rstdLocal, rstdLocal, epsilon, calcRowNum);
        AscendC::PipeBarrier<PIPE_V>();
        Sqrt(rstdLocal, rstdLocal, calcRowNum);
        Duplicate(tmpLocal, ONE, calcRowNum);
        AscendC::PipeBarrier<PIPE_V>();
        Div(rstdLocal, tmpLocal, rstdLocal, calcRowNum);
        AscendC::PipeBarrier<PIPE_V>();
        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(iOuter, calcRowNum);

        // 3. BroadCast rstd
        const uint32_t srcShape[2] = {calcRowNum, 1};
        const uint32_t dstShape[2] = {calcRowNum, numColAlign};
        auto sharedTmpLocal = tmpLocal.ReinterpretCast<uint8_t>();
        BroadCast<float, DIM_NUM, 1>(sqx, rstdLocal, dstShape, srcShape, sharedTmpLocal);
        AscendC::PipeBarrier<PIPE_V>();

        // 4. Y = x * rstd * gamma
        if constexpr (IsSame<T, float>::value) {  // fp32 use inQueueX store x
            Mul(sqx, xLocal, sqx, elementNum);
            inQueueX.FreeTensor(xLocal);
        } else {  // fp16/bf16 use xFp32Buf store x
            Mul(sqx, xFp32, sqx, elementNum);
        }
        AscendC::PipeBarrier<PIPE_V>();
        ComputeMulGammaCast(gammaLocal, elementNum);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOutY(uint32_t progress, uint32_t calc_row_num)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        if (isNumColAlign) {
            DataCopyCustom<T>(yGm[progress], yLocal, calc_row_num * numCol);
        } else {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            // only support v220
            DataCopyParams copyParams;
            copyParams.blockLen = numCol * sizeof(T);
            copyParams.blockCount = calc_row_num;
            DataCopyPad(yGm[progress], yLocal, copyParams);
#endif
        }
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint32_t outer_progress, uint32_t num)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        DataCopyCustom<float>(rstdGm[outer_progress * rowFactor], rstdLocal, num);
#endif
        outQueueRstd.FreeTensor(rstdLocal);
    }

private:
    TPipe pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueGamma;
    // create queues for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY, outQueueRstd;

    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> sqxBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    GlobalTensor<T> xGm;
    GlobalTensor<T_GAMMA> gammaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> rstdGm;

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
#endif  // RMS_NORM_MERGE_N_H_
