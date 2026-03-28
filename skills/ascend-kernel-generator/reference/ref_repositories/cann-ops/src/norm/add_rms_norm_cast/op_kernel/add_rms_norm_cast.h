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
 * \file add_rms_norm_cast.h
 * \brief
 */
#ifndef ADD_RMS_NORM_CAST_H_
#define ADD_RMS_NORM_CAST_H_
#include "rms_norm_base.h"

using namespace AscendC;

template <typename T>
class KernelAddRmsNormCast {
public:
    __aicore__ inline KernelAddRmsNormCast(TPipe *pipe)
    {
        Ppipe = pipe;
    }
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y1, GM_ADDR y2, GM_ADDR rstd, GM_ADDR x,
        GM_ADDR workspace, const AddRMSNormCastTilingData *tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        this->numRow = tiling->num_row;
        this->numCol = tiling->num_col;
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
        }
        // get start index for current core, core parallel
        x1Gm.SetGlobalBuffer((__gm__ T *)x1 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ T *)gamma, numCol);
        y1Gm.SetGlobalBuffer((__gm__ float *)y1 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        y2Gm.SetGlobalBuffer((__gm__ T *)y2 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        rstdGm.SetGlobalBuffer((__gm__ float *)rstd + blockIdx_ * blockFactor, blockFactor);
        xGm.SetGlobalBuffer((__gm__ T *)x + blockIdx_ * blockFactor * numCol, rowWork * numCol);

        // pipe alloc memory to queue, the unit is Bytes
        Ppipe->InitBuffer(inQueueX, BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(outQueueY, BUFFER_NUM, ubFactor * sizeof(T));
        Ppipe->InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * sizeof(float));

        if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
            Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        }
        Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();

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
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
            uint32_t gm_bias = (i_o * rowFactor + i_i) * numCol;
            CopyIn(gm_bias);
            Compute(i_i, gammaLocal, rstdLocal, gm_bias);
            CopyOutY(gm_bias);
        }
        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(i_o, calc_row_num);
    }

private:
    __aicore__ inline void CopyIn(uint32_t gm_bias)
    {
        LocalTensor<T> x1Local_in = inQueueX.AllocTensor<T>();
        LocalTensor<T> x2Local = sqxBuf.Get<T>();
        LocalTensor<T> xLocal = outQueueY.AllocTensor<T>();
        if constexpr (IsSame<T, half>::value || IsSame<T, bfloat16_t>::value) {
            x2Local = x2Local[ubFactor];
        }

        DataCopyCustom<T>(x1Local_in, x1Gm[gm_bias], numCol);
        DataCopyCustom<T>(x2Local, x2Gm[gm_bias], numCol);
        inQueueX.EnQue(x1Local_in);
        auto x1Local = inQueueX.DeQue<T>();

        PipeBarrier<PIPE_ALL>();
        if constexpr (IsSame<T, half>::value) {
            LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
            Add(xLocal, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();
            Cast(x1_fp32, xLocal, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
        } else {
            LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
            LocalTensor<float> x2_fp32 = sqxBuf.Get<float>();
            Cast(x1_fp32, x1Local, RoundMode::CAST_NONE, numCol);
            Cast(x2_fp32, x2Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(x1_fp32, x1_fp32, x2_fp32, numCol);
            PipeBarrier<PIPE_V>();
            Cast(xLocal, x1_fp32, RoundMode::CAST_RINT, numCol);
            PipeBarrier<PIPE_V>();
        }
        inQueueX.FreeTensor(x1Local);
        // CopyOut x1 + x2
        outQueueY.EnQue(xLocal);
        auto x_out = outQueueY.DeQue<T>();
        DataCopyCustom<T>(xGm[gm_bias], x_out, numCol);
        outQueueY.FreeTensor(x_out);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void Compute(
        uint32_t inner_progress, LocalTensor<bfloat16_t> gammaLocal, LocalTensor<float> rstdLocal, uint32_t progress)
    {
        LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();

        Mul(sqx, x_fp32, x_fp32, numCol);
        PipeBarrier<PIPE_V>();

        Muls(sqx, sqx, avgFactor, numCol);
        PipeBarrier<PIPE_V>();
        ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
        PipeBarrier<PIPE_V>();

        Adds(sqx, sqx, epsilon, 1);
        PipeBarrier<PIPE_V>();

        Sqrt(sqx, sqx, 1);
        Duplicate(reduce_buf_local, ONE, 1);
        PipeBarrier<PIPE_V>();
        Div(sqx, reduce_buf_local, sqx, 1);
        PipeBarrier<PIPE_V>();
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float rstdValue = sqx.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        rstdLocal.SetValue(inner_progress, rstdValue);
        PipeBarrier<PIPE_V>();
        Muls(x_fp32, x_fp32, rstdValue, numCol);
        PipeBarrier<PIPE_V>();
        LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();
        Cast(sqx, gammaLocal, RoundMode::CAST_NONE, numCol);  // gamma_fp32 reuse sqx
        PipeBarrier<PIPE_V>();
        Mul(x_fp32, x_fp32, sqx, numCol);
        PipeBarrier<PIPE_V>();
        Cast(yLocal, x_fp32, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();

        event_t event_v_mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(event_v_mte);
        WaitFlag<HardEvent::V_MTE2>(event_v_mte);

        outQueueY.EnQue<bfloat16_t>(yLocal);
        Cast(x_fp32, yLocal, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        event_t event_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event_v_mte3);
        WaitFlag<HardEvent::V_MTE3>(event_v_mte3);
        DataCopyCustom<float>(y1Gm[progress], x_fp32, numCol);
        event_t event_mte3_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event_mte3_v);
        WaitFlag<HardEvent::MTE3_V>(event_mte3_v);
    }

    __aicore__ inline void Compute(
        uint32_t inner_progress, LocalTensor<half> gammaLocal, LocalTensor<float> rstdLocal, uint32_t progress)
    {
        LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();

        Mul(sqx, x_fp32, x_fp32, numCol);
        PipeBarrier<PIPE_V>();

        Muls(sqx, sqx, avgFactor, numCol);
        PipeBarrier<PIPE_V>();

        ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
        PipeBarrier<PIPE_V>();

        Adds(sqx, sqx, epsilon, 1);
        PipeBarrier<PIPE_V>();

        Sqrt(sqx, sqx, 1);
        Duplicate(reduce_buf_local, ONE, 1);
        PipeBarrier<PIPE_V>();
        Div(sqx, reduce_buf_local, sqx, 1);
        PipeBarrier<PIPE_V>();
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float rstdValue = sqx.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        rstdLocal.SetValue(inner_progress, rstdValue);
        PipeBarrier<PIPE_V>();
        Muls(x_fp32, x_fp32, rstdValue, numCol);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        Cast(yLocal, x_fp32, RoundMode::CAST_NONE, numCol);

        event_t event_v_mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(event_v_mte);
        WaitFlag<HardEvent::V_MTE2>(event_v_mte);

        PipeBarrier<PIPE_V>();
        Mul(yLocal, gammaLocal, yLocal, numCol);
        PipeBarrier<PIPE_V>();
        outQueueY.EnQue<half>(yLocal);
        Cast(x_fp32, yLocal, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        event_t event_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event_v_mte3);
        WaitFlag<HardEvent::V_MTE3>(event_v_mte3);
        DataCopyCustom<float>(y1Gm[progress], x_fp32, numCol);
        event_t event_mte3_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event_mte3_v);
        WaitFlag<HardEvent::MTE3_V>(event_mte3_v);
    }

    __aicore__ inline void CopyOutY(uint32_t progress)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyCustom<T>(y2Gm[progress], yLocal, numCol);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint32_t outer_progress, uint32_t num)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
#if __CCE_AICORE__ == 220
        DataCopyCustom<float>(rstdGm[outer_progress * rowFactor], rstdLocal, num);
#endif
        outQueueRstd.FreeTensor(rstdLocal);
    }

private:
    TPipe *Ppipe = nullptr;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGamma;
    // create queues for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueRstd;

    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> sqxBuf;
    TBuf<TPosition::VECCALC> reduceFp32Buf;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<float> y1Gm;
    GlobalTensor<T> y2Gm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<T> xGm;

    uint32_t numRow;
    uint32_t numCol;
    uint32_t blockFactor;  // number of calculations rows on each core
    uint32_t rowFactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    int32_t blockIdx_;
    uint32_t rowWork = 1;
};
#endif  // ADD_RMS_NORM_CAST_H_