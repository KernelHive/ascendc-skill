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
 * \file add_rms_norm_quant_split_d.h
 * \brief
 */
#ifndef ADD_RMS_NORM_QUANT_SPLIT_D_H_
#define ADD_RMS_NORM_QUANT_SPLIT_D_H_
#include "rms_norm_base.h"

using namespace AscendC;

template <typename TX, typename TScale, typename TOffset>
class KernelAddRmsNormQuantSplitD {
public:
    __aicore__ inline KernelAddRmsNormQuantSplitD(TPipe *pipe)
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

        // pipe alloc memory to queue, the unit is Bytes.
        // We need 2 buffers here for both x1 and x2.
        Ppipe->InitBuffer(inQueueX, BUFFER_NUM, 2 * ubFactor * sizeof(TX));
        Ppipe->InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(TX));
        Ppipe->InitBuffer(outQueueY1, BUFFER_NUM, ubFactor * sizeof(TX));

        Ppipe->InitBuffer(scales1Buf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(zeroPoints1Buf, ubFactor * sizeof(int32_t));
        if constexpr (IsSame<TX, half>::value || IsSame<TX, bfloat16_t>::value) {
            Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        }
        Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(sumBuf, rowFactor * NUM_PER_BLK_FP32 * sizeof(float));
        Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
        Ppipe->InitBuffer(rstdBuf, rowFactor * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t iOMax = CeilDiv(rowWork, rowFactor);
        uint32_t rowTail = rowWork - (iOMax - 1) * rowFactor;
        uint32_t jMax = CeilDiv(numCol, ubFactor);
        uint32_t colTail = numCol - (jMax - 1) * ubFactor;
        for (uint32_t iO = 0; iO < iOMax - 1; iO++) {
            SubProcess(iO, rowFactor, jMax, colTail);
        }
        SubProcess(iOMax - 1, rowTail, jMax, colTail);
    }

    __aicore__ inline void SubProcess(uint32_t iO, uint32_t calcRowNum, uint32_t jMax, uint32_t colTail)
    {
        LocalTensor<float> sumLocal = sumBuf.Get<float>();

        LocalTensor<float> rstdLocal = rstdBuf.Get<float>();
        Duplicate(rstdLocal, (float)0.0, calcRowNum);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < jMax - 1; j++) {
            ComputeFormer(iO, calcRowNum, j, rstdLocal, sumLocal, ubFactor);
        }
        // do tail
        ComputeFormer(iO, calcRowNum, jMax - 1, rstdLocal, sumLocal, colTail);
        ComputeRstd(rstdLocal, calcRowNum);

        for (uint32_t j = 0; j < jMax - 1; j++) {
            ComputeLatter(iO, calcRowNum, j, rstdLocal, ubFactor);
        }
        ComputeLatter(iO, calcRowNum, jMax - 1, rstdLocal, colTail);
    }

private:
    __aicore__ inline void CopyInAndAdd(uint32_t iIdx, uint32_t jIdx, uint32_t num)
    {
        LocalTensor<TX> x1x2In = inQueueX.AllocTensor<TX>();
        LocalTensor<TX> x1In = x1x2In[0];
        LocalTensor<TX> x2In = x1x2In[ubFactor];
        DataCopyCustom<TX>(x1In, x1Gm[iIdx * numCol + jIdx * ubFactor], num);
        DataCopyCustom<TX>(x2In, x2Gm[iIdx * numCol + jIdx * ubFactor], num);
        inQueueX.EnQue(x1x2In);
        LocalTensor<TX> x1x2Local = inQueueX.DeQue<TX>();

        auto x1Local = x1x2Local[0];
        auto x2Local = x1x2Local[ubFactor];

        LocalTensor<TX> xLocal = outQueueY1.AllocTensor<TX>();

        if constexpr (IsSame<TX, half>::value) {
            LocalTensor<float> x1Fp32Local = xFp32Buf.Get<float>();

            Add(xLocal, x1Local, x2Local, num);
            PipeBarrier<PIPE_V>();
            Cast(x1Fp32Local, xLocal, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
            // x1+x2 saved in x1Fp32Local
        } else if constexpr (IsSame<TX, bfloat16_t>::value) {
            LocalTensor<float> x1Fp32Local = xFp32Buf.Get<float>();
            LocalTensor<float> x2Fp32Local = x1x2Local.template ReinterpretCast<float>();

            Cast(x1Fp32Local, x1Local, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
            Cast(x2Fp32Local, x2Local, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();

            Add(x1Fp32Local, x1Fp32Local, x2Fp32Local, num);
            PipeBarrier<PIPE_V>();
            Cast(xLocal, x1Fp32Local, RoundMode::CAST_RINT, num);
            PipeBarrier<PIPE_V>();
            // x1+x2 saved in x1Fp32Local
        } else {
            Add(x1Local, x1Local, x2Local, num);
            PipeBarrier<PIPE_V>();
            Adds(xLocal, x1Local, (float)0.0, num);
            // x1+x2 saved in inQueueX
        }
        inQueueX.FreeTensor(x1x2Local);

        // copy out to workspace && x_out
        outQueueY1.EnQue(xLocal);
        auto x_out = outQueueY1.DeQue<TX>();
        DataCopyCustom<TX>(xGm[iIdx * numCol + jIdx * ubFactor], x_out, num);
        outQueueY1.FreeTensor(x_out);
    }

    __aicore__ inline void ComputeFormer(uint32_t iOIdx, uint32_t calcRowNum, uint32_t jIdx,
        LocalTensor<float> &rstdLocal, LocalTensor<float> &sumLocal, uint32_t num)
    {
        for (uint32_t i_i = 0; i_i < calcRowNum; i_i++) {
            CopyInAndAdd(iOIdx * rowFactor + i_i, jIdx, num);
            ComputeSum(i_i, sumLocal, num);
        }
        BlockReduceSumFP32(sumLocal, sumLocal, calcRowNum * NUM_PER_BLK_FP32);
        Add(rstdLocal, rstdLocal, sumLocal, calcRowNum);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeSum(uint32_t iIIdx, LocalTensor<float> &sumLocal, uint32_t num)
    {
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
        if constexpr (IsSame<TX, half>::value || IsSame<TX, bfloat16_t>::value) {
            LocalTensor<float> xFp32Local = xFp32Buf.Get<float>();
            PipeBarrier<PIPE_V>();
            Mul(sqx, xFp32Local, xFp32Local, num);
        } else {
            LocalTensor<TX> xLocal = inQueueX.AllocTensor<float>();
            PipeBarrier<PIPE_V>();
            Mul(sqx, xLocal, xLocal, num);
            inQueueX.FreeTensor(xLocal);
        }
        PipeBarrier<PIPE_V>();
        Muls(sqx, sqx, avgFactor, num);
        PipeBarrier<PIPE_V>();
        // 8 means 8 fp32 pre block
        ReduceSumFP32ToBlock(sumLocal[iIIdx * 8], sqx, reduce_buf_local, num);
    }

    __aicore__ inline void ComputeRstd(LocalTensor<float> rstdLocal, uint32_t num)
    {
        LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
        Adds(rstdLocal, rstdLocal, epsilon, num);
        PipeBarrier<PIPE_V>();
        Sqrt(rstdLocal, rstdLocal, num);
        Duplicate(reduce_buf_local, ONE, num);
        PipeBarrier<PIPE_V>();
        Div(rstdLocal, reduce_buf_local, rstdLocal, num);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeLatter(
        uint32_t iOIdx, uint32_t calcRowNum, uint32_t jIdx, LocalTensor<float> &rstdLocal, uint32_t num)
    {
        CopyInGamma(jIdx, num);
        LocalTensor<TX> gammaLocal = inQueueGamma.DeQue<TX>();
        for (uint32_t i_i = 0; i_i < calcRowNum; i_i++) {
            CopyInAndAdd(iOIdx * rowFactor + i_i, jIdx, num);
            ComputeY(i_i, gammaLocal, rstdLocal, num);
            CopyOutY(iOIdx * rowFactor + i_i, jIdx, num);
        }
        inQueueGamma.FreeTensor(gammaLocal);
    }

    __aicore__ inline void CopyInGamma(uint32_t jIdx, uint32_t num)
    {
        LocalTensor<float> scales1Local = scales1Buf.Get<float>();
        if constexpr (IsSame<TScale, float>::value) {
            DataCopyCustom<float>(scales1Local, scales1Gm[jIdx * ubFactor], num);
        } else {  // bfloat16
            LocalTensor<bfloat16_t> scales1Bf16 = scales1Buf.Get<bfloat16_t>()[ubFactor];
            DataCopyCustom<bfloat16_t>(scales1Bf16, scales1Gm[jIdx * ubFactor], num);
            event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMte2V);
            WaitFlag<HardEvent::MTE2_V>(eventMte2V);
            Cast(scales1Local, scales1Bf16, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
        }
        if (hasZeroPoints1) {
            LocalTensor<float> zeroPoints1Fp32 = zeroPoints1Buf.Get<float>();
            if constexpr (IsSame<TOffset, int32_t>::value) {
                LocalTensor<int32_t> zeroPoints1Int32 = zeroPoints1Buf.Get<int32_t>();
                DataCopyCustom<int32_t>(zeroPoints1Int32, zeroPoints1Gm[jIdx * ubFactor], num);
                event_t eventMte2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventMte2V2);
                WaitFlag<HardEvent::MTE2_V>(eventMte2V2);
                Cast(zeroPoints1Fp32, zeroPoints1Int32, RoundMode::CAST_NONE, num);
                PipeBarrier<PIPE_V>();
            } else {
                LocalTensor<bfloat16_t> zeroPoints1Bf16 = zeroPoints1Buf.Get<bfloat16_t>()[ubFactor];
                DataCopyCustom<bfloat16_t>(zeroPoints1Bf16, zeroPoints1Gm[jIdx * ubFactor], num);
                event_t eventMte2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventMte2V2);
                WaitFlag<HardEvent::MTE2_V>(eventMte2V2);
                Cast(zeroPoints1Fp32, zeroPoints1Bf16, RoundMode::CAST_NONE, num);
                PipeBarrier<PIPE_V>();
            }
        }
        LocalTensor<TX> gammaLocal = inQueueGamma.AllocTensor<TX>();
        DataCopyCustom<TX>(gammaLocal, gammaGm[jIdx * ubFactor], num);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void ComputeY(
        uint32_t iIIdx, LocalTensor<TX> &gammaLocal, LocalTensor<float> &rstdLocal, uint32_t num)
    {
        LocalTensor<float> xFp32Local = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        float rstdValue = rstdLocal.GetValue(iIIdx);
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        PipeBarrier<PIPE_V>();
        Muls(xFp32Local, xFp32Local, rstdValue, num);
        PipeBarrier<PIPE_V>();

        if constexpr (IsSame<TX, half>::value) {
            LocalTensor<half> xFp16Cast = sqxBuf.Get<half>();
            Cast(xFp16Cast, xFp32Local, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
            Mul(xFp16Cast, gammaLocal, xFp16Cast, num);
            PipeBarrier<PIPE_V>();
            Cast(xFp32Local, xFp16Cast, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
        } else {  // bfloat16
            Cast(sqx, gammaLocal, RoundMode::CAST_NONE, num);
            PipeBarrier<PIPE_V>();
            Mul(xFp32Local, sqx, xFp32Local, num);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> scales1Local = scales1Buf.Get<float>();
        Div(xFp32Local, xFp32Local, scales1Local, num);
        PipeBarrier<PIPE_V>();

        if (hasZeroPoints1) {
            LocalTensor<float> zeroPoints1Fp32 = zeroPoints1Buf.Get<float>();
            Add(xFp32Local, xFp32Local, zeroPoints1Fp32, num);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<int8_t> y1Local = outQueueY1.AllocTensor<int8_t>();
        RoundFloat2Int8(y1Local, xFp32Local, num);
        outQueueY1.EnQue<int8_t>(y1Local);
    }

    __aicore__ inline void CopyOutY(uint32_t iIdx, uint32_t jIdx, uint32_t num)
    {
        LocalTensor<int8_t> yLocal = outQueueY1.DeQue<int8_t>();
        PipeBarrier<PIPE_ALL>();
        DataCopyCustom<int8_t>(y1Gm[iIdx * numCol + jIdx * ubFactor], yLocal, num);
        PipeBarrier<PIPE_ALL>();
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
    TBuf<TPosition::VECCALC> sumBuf;
    TBuf<TPosition::VECCALC> reduceFp32Buf;
    TBuf<TPosition::VECCALC> rstdBuf;

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

    int tempbufNum;
};
#endif  // ADD_RMS_NORM_QUANT_SPLIT_D_H_