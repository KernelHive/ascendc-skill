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
 * \file add_rms_norm_quant.h
 * \brief
 */
#ifndef ADD_RMS_NORM_QUANT_H_
#define ADD_RMS_NORM_QUANT_H_
#include "rms_norm_base.h"

using namespace AscendC;

template <typename TX, typename TScale, typename TOffset>
class KernelAddRmsNormQuant {
public:
    __aicore__ inline KernelAddRmsNormQuant(TPipe *pipe)
    {
        Ppipe = pipe;
    }
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2,
        GM_ADDR zero_points1, GM_ADDR zero_points2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x,
        const AddRMSNormQuantTilingData *tilingData)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        this->numRow = tilingData->numRow;
        this->numCol = tilingData->numCol;
        this->blockFactor = tilingData->blockFactor;
        this->rowFactor = tilingData->rowFactor;
        this->ubFactor = tilingData->ubFactor;
        this->epsilon = tilingData->epsilon;
        this->avgFactor = (float)1.0 / numCol;
        this->hasZeroPoints1 = tilingData->hasZeroPoints1;

        blockIdx_ = GetBlockIdx();
        if (blockIdx_ < GetBlockNum() - 1) {
            this->rowWork = blockFactor;
        } else if (blockIdx_ == GetBlockNum() - 1) {
            this->rowWork = numRow - (GetBlockNum() - 1) * blockFactor;
        } else {
        }
        // get start index for current core, core parallel
        x1Gm.SetGlobalBuffer((__gm__ TX *)x1 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        x2Gm.SetGlobalBuffer((__gm__ TX *)x2 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ TX *)gamma, numCol);
        scales1Gm.SetGlobalBuffer((__gm__ TScale *)scales1, numCol);
        if (hasZeroPoints1) {
            zeroPoints1Gm.SetGlobalBuffer((__gm__ TOffset *)zero_points1, numCol);
        }
        y1Gm.SetGlobalBuffer((__gm__ int8_t *)y1 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        y2Gm.SetGlobalBuffer((__gm__ int8_t *)y2 + blockIdx_ * blockFactor * numCol, rowWork * numCol);
        xGm.SetGlobalBuffer((__gm__ TX *)x + blockIdx_ * blockFactor * numCol, rowWork * numCol);

        // pipe alloc memory to queue, the unit is Bytes
        Ppipe->InitBuffer(inQueueX, BUFFER_NUM, ubFactor * sizeof(TX));
        Ppipe->InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(TX));
        Ppipe->InitBuffer(outQueueY1, BUFFER_NUM, ubFactor * sizeof(TX));

        Ppipe->InitBuffer(scales1Buf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(zeroPoints1Buf, ubFactor * sizeof(int32_t));
        if constexpr (IsSame<TX, half>::value || IsSame<TX, bfloat16_t>::value) {
            Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        }
        Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<TX> gammaLocal = inQueueGamma.DeQue<TX>();

        uint32_t iOMax = CeilDiv(rowWork, rowFactor);
        uint32_t rowTail = rowWork - (iOMax - 1) * rowFactor;

        for (uint32_t iO = 0; iO < iOMax - 1; iO++) {
            SubProcess(iO, rowFactor, gammaLocal);
        }
        SubProcess(iOMax - 1, rowTail, gammaLocal);
        inQueueGamma.FreeTensor(gammaLocal);
    }

    __aicore__ inline void SubProcess(uint32_t iO, uint32_t calcRowNum, LocalTensor<TX> &gammaLocal)
    {
        for (uint32_t iI = 0; iI < calcRowNum; iI++) {
            uint32_t gmBias = (iO * rowFactor + iI) * numCol;
            CopyIn(gmBias);
            Compute(iI, gammaLocal);
            CopyOutY(gmBias);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t gmBias)
    {
        LocalTensor<TX> x1LocalIn = inQueueX.AllocTensor<TX>();
        LocalTensor<TX> x2Local = sqxBuf.Get<TX>();
        LocalTensor<TX> xLocal = outQueueY1.AllocTensor<TX>();

        if constexpr (IsSame<TX, half>::value || IsSame<TX, bfloat16_t>::value) {
            x2Local = x2Local[ubFactor];
        }

        DataCopyCustom<TX>(x1LocalIn, x1Gm[gmBias], numCol);
        DataCopyCustom<TX>(x2Local, x2Gm[gmBias], numCol);
        inQueueX.EnQue(x1LocalIn);
        auto x1Local = inQueueX.DeQue<TX>();

        if constexpr (IsSame<TX, half>::value) {
            LocalTensor<float> x1Fp32Local = xFp32Buf.Get<float>();
            Add(xLocal, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();
            Cast(x1Fp32Local, xLocal, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
        } else if constexpr (IsSame<TX, bfloat16_t>::value) {
            LocalTensor<float> x1Fp32Local = xFp32Buf.Get<float>();
            LocalTensor<float> x2_fp32 = sqxBuf.Get<float>();
            Cast(x1Fp32Local, x1Local, RoundMode::CAST_NONE, numCol);
            Cast(x2_fp32, x2Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(x1Fp32Local, x1Fp32Local, x2_fp32, numCol);
            PipeBarrier<PIPE_V>();
            Cast(xLocal, x1Fp32Local, RoundMode::CAST_RINT, numCol);
            PipeBarrier<PIPE_V>();
        } else {
            Add(x1Local, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();
            Adds(xLocal, x1Local, (float)0, numCol);
        }
        inQueueX.FreeTensor(x1Local);

        // CopyOut x1 + x2
        outQueueY1.EnQue(xLocal);
        auto xOut = outQueueY1.DeQue<TX>();
        DataCopyCustom<TX>(xGm[gmBias], xOut, numCol);
        outQueueY1.FreeTensor(xOut);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<float> scales1Local = scales1Buf.Get<float>();
        if constexpr (IsSame<TScale, float>::value) {
            DataCopyCustom<float>(scales1Local, scales1Gm, numCol);
        } else {  // bfloat16
            LocalTensor<bfloat16_t> scales1Bf16 = scales1Buf.Get<bfloat16_t>()[ubFactor];
            DataCopyCustom<bfloat16_t>(scales1Bf16, scales1Gm, numCol);
            event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMte2V);
            WaitFlag<HardEvent::MTE2_V>(eventMte2V);
            Cast(scales1Local, scales1Bf16, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
        }

        if (hasZeroPoints1) {
            LocalTensor<float> zeroPoints1Fp32 = zeroPoints1Buf.Get<float>();
            if constexpr (IsSame<TOffset, int32_t>::value) {
                LocalTensor<int32_t> zeroPoints1Int32 = zeroPoints1Buf.Get<int32_t>();
                DataCopyCustom<int32_t>(zeroPoints1Int32, zeroPoints1Gm, numCol);
                event_t eventMte2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventMte2V2);
                WaitFlag<HardEvent::MTE2_V>(eventMte2V2);
                Cast(zeroPoints1Fp32, zeroPoints1Int32, RoundMode::CAST_NONE, numCol);
                PipeBarrier<PIPE_V>();
            } else {
                LocalTensor<bfloat16_t> zeroPoints1Bf16 = zeroPoints1Buf.Get<bfloat16_t>()[ubFactor];
                DataCopyCustom<bfloat16_t>(zeroPoints1Bf16, zeroPoints1Gm, numCol);
                event_t eventMte2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventMte2V2);
                WaitFlag<HardEvent::MTE2_V>(eventMte2V2);
                Cast(zeroPoints1Fp32, zeroPoints1Bf16, RoundMode::CAST_NONE, numCol);
                PipeBarrier<PIPE_V>();
            }
        }
        LocalTensor<TX> gammaLocal = inQueueGamma.AllocTensor<TX>();
        DataCopyCustom<TX>(gammaLocal, gammaGm, numCol);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void Compute(uint32_t inner_progress, LocalTensor<TX> gammaLocal)
    {
        LocalTensor<float> xFp32Local = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduceLocal = reduceFp32Buf.Get<float>();

        Mul(sqx, xFp32Local, xFp32Local, numCol);
        PipeBarrier<PIPE_V>();

        Muls(sqx, sqx, avgFactor, numCol);
        PipeBarrier<PIPE_V>();

        ReduceSumCustom(sqx, sqx, reduceLocal, numCol);
        PipeBarrier<PIPE_V>();

        Adds(sqx, sqx, epsilon, 1);
        PipeBarrier<PIPE_V>();

        Sqrt(sqx, sqx, 1);
        Duplicate(reduceLocal, ONE, 1);
        PipeBarrier<PIPE_V>();
        Div(sqx, reduceLocal, sqx, 1);
        PipeBarrier<PIPE_V>();
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float rstdValue = sqx.GetValue(0);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        Muls(xFp32Local, xFp32Local, rstdValue, numCol);
        PipeBarrier<PIPE_V>();

        if constexpr (IsSame<TX, half>::value) {
            LocalTensor<half> xFp16Cast = sqxBuf.Get<half>();
            Cast(xFp16Cast, xFp32Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Mul(xFp16Cast, gammaLocal, xFp16Cast, numCol);
            PipeBarrier<PIPE_V>();
            Cast(xFp32Local, xFp16Cast, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
        } else {  // bfloat16
            Cast(sqx, gammaLocal, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Mul(xFp32Local, sqx, xFp32Local, numCol);
            PipeBarrier<PIPE_V>();
        }

        event_t event_v_mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(event_v_mte);
        WaitFlag<HardEvent::V_MTE2>(event_v_mte);

        LocalTensor<float> scales1Local = scales1Buf.Get<float>();
        Div(xFp32Local, xFp32Local, scales1Local, numCol);
        PipeBarrier<PIPE_V>();

        if (hasZeroPoints1) {
            LocalTensor<float> zeroPoints1Fp32 = zeroPoints1Buf.Get<float>();
            Add(xFp32Local, xFp32Local, zeroPoints1Fp32, numCol);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<int8_t> y1Local = outQueueY1.AllocTensor<int8_t>();
        RoundFloat2Int8(y1Local, xFp32Local, numCol);
        outQueueY1.EnQue<int8_t>(y1Local);
    }

    __aicore__ inline void CopyOutY(uint32_t progress)
    {
        LocalTensor<int8_t> yLocal = outQueueY1.DeQue<int8_t>();
        DataCopyCustom<int8_t>(y1Gm[progress], yLocal, numCol);
        outQueueY1.FreeTensor(yLocal);
    }

private:
    TPipe *Ppipe = nullptr;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueGamma;
    // create queues for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY1;

    TBuf<TPosition::VECCALC> scales1Buf;
    TBuf<TPosition::VECCALC> zeroPoints1Buf;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> sqxBuf;
    TBuf<TPosition::VECCALC> reduceFp32Buf;
    GlobalTensor<TX> x1Gm;
    GlobalTensor<TX> x2Gm;
    GlobalTensor<TX> gammaGm;
    GlobalTensor<TScale> scales1Gm;
    GlobalTensor<TScale> scales2Gm;
    GlobalTensor<TOffset> zeroPoints1Gm;
    GlobalTensor<TOffset> zeroPoints2Gm;
    GlobalTensor<int8_t> y1Gm;
    GlobalTensor<int8_t> y2Gm;
    GlobalTensor<TX> xGm;

    uint32_t numRow;
    uint32_t numCol;
    uint32_t blockFactor;  // number of calculations rows on each core
    uint32_t rowFactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    uint32_t hasZeroPoints1;
    int32_t blockIdx_;
    uint32_t rowWork = 1;
};
#endif  // ADD_RMS_NORM_QUANT_H_