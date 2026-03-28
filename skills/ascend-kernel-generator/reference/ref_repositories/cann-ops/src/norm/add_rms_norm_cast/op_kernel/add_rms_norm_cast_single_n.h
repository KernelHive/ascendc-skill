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
 * \file add_rms_norm_cast_single_n.h
 * \brief
 */
#ifndef ADD_RMS_NORM_CAST_SINGLE_N_H_
#define ADD_RMS_NORM_CAST_SINGLE_N_H_
#include "rms_norm_base.h"

using namespace AscendC;

template <typename T>
class KernelAddRmsNormCastSingleN {
public:
    __aicore__ inline KernelAddRmsNormCastSingleN(TPipe *pipe)
    {
        Ppipe = pipe;
    }
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y1, GM_ADDR y2, GM_ADDR rstd, GM_ADDR x,
        GM_ADDR workspace, const AddRMSNormCastTilingData *tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");

        this->numCol = tiling->num_col;
        this->blockFactor = 1;  // in this case, blockFactor = 1
        this->ubFactor = tiling->ub_factor;
        this->epsilon = tiling->epsilon;
        this->avgFactor = (numCol != 0) ? (float)1.0 / numCol : 0;

        this->rowWork = 1;
        blockIdx_ = GetBlockIdx();
        // get start index for current core, core parallel
        x1Gm.SetGlobalBuffer((__gm__ T *)x1 + blockIdx_ * numCol, numCol);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2 + blockIdx_ * numCol, numCol);
        gammaGm.SetGlobalBuffer((__gm__ T *)gamma, numCol);
        y1Gm.SetGlobalBuffer((__gm__ float *)y1 + blockIdx_ * numCol, numCol);
        y2Gm.SetGlobalBuffer((__gm__ T *)y2 + blockIdx_ * numCol, numCol);
        rstdGm.SetGlobalBuffer((__gm__ float *)rstd + blockIdx_, 1);
        xGm.SetGlobalBuffer((__gm__ T *)x + blockIdx_ * numCol, numCol);

        Ppipe->InitBuffer(unitBuf, 195584);  // (192 - 1) * 1024 byte
    }

    __aicore__ inline void Process()
    {
        if constexpr (IsSame<T, half>::value) {
            ProcessFp16();
        } else {
            ProcessBf16();
        }
    }

private:
    __aicore__ inline void ProcessFp16()
    {
        LocalTensor<float> ubLocal = unitBuf.Get<float>();
        LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
        LocalTensor<T> x1Local = xLocal[0];
        LocalTensor<T> x2Local = xLocal[ubFactor];
        LocalTensor<float> xFp32Local = ubLocal[ubFactor];
        LocalTensor<float> sqxLocal = ubLocal[ubFactor * 2];
        LocalTensor<float> tmpLocal = ubLocal[ubFactor * 3];

        DataCopyCustom<T>(x1Local, x1Gm, numCol);
        event_t eventMTE2V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V1);
        DataCopyCustom<T>(x2Local, x2Gm, numCol);
        event_t eventMTE2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V1);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
        Add(x1Local, x1Local, x2Local, numCol);
        PipeBarrier<PIPE_V>();

        // copy gamma
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);

        DataCopyCustom<T>(x2Local, gammaGm, numCol);  // gammaLocal use x2Local
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2);

        // copy x out
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<T>(xGm, x1Local, numCol);
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);

        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        Mul(sqxLocal, xFp32Local, xFp32Local, numCol);
        PipeBarrier<PIPE_V>();
        Muls(sqxLocal, sqxLocal, avgFactor, numCol);
        PipeBarrier<PIPE_V>();
        ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, numCol);
        PipeBarrier<PIPE_V>();
        Adds(sqxLocal, sqxLocal, epsilon, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sqxLocal, sqxLocal, 1);
        Duplicate(tmpLocal, ONE, 1);
        PipeBarrier<PIPE_V>();
        Div(sqxLocal, tmpLocal, sqxLocal, 1);
        PipeBarrier<PIPE_V>();

        // copyout rstd
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<float>(rstdGm, sqxLocal, 1);
#endif
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float rstdValue = sqxLocal.GetValue(0);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);

        Muls(xFp32Local, xFp32Local, rstdValue, numCol);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
        Cast(x1Local, xFp32Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
        Mul(x1Local, x1Local, x2Local, numCol);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<T>(y2Gm, x1Local, numCol);
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<float>(y1Gm, xFp32Local, numCol);
    }

    __aicore__ inline void ProcessBf16()
    {
        LocalTensor<float> ubLocal = unitBuf.Get<float>();
        LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
        LocalTensor<T> x1Local = xLocal[0];
        LocalTensor<T> x2Local = xLocal[ubFactor];
        LocalTensor<float> xFp32Local = ubLocal[ubFactor];
        LocalTensor<float> sqxLocal = ubLocal[ubFactor * 2];
        LocalTensor<float> tmpLocal = ubLocal[ubFactor * 3];

        DataCopyCustom<T>(x1Local, x1Gm, numCol);
        event_t eventMTE2V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V1);
        DataCopyCustom<T>(x2Local, x2Gm, numCol);
        event_t eventMTE2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V1);
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
        Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        Add(xFp32Local, xFp32Local, sqxLocal, numCol);
        PipeBarrier<PIPE_V>();
        Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();
        // copy gamma
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);

        DataCopyCustom<T>(x2Local, gammaGm, numCol);  // gammaLocal use x2Local
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2);

        // copy x out
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<T>(xGm, x1Local, numCol);
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);

        Mul(sqxLocal, xFp32Local, xFp32Local, numCol);
        PipeBarrier<PIPE_V>();
        Muls(sqxLocal, sqxLocal, avgFactor, numCol);
        PipeBarrier<PIPE_V>();
        ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, numCol);
        PipeBarrier<PIPE_V>();
        Adds(sqxLocal, sqxLocal, epsilon, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sqxLocal, sqxLocal, 1);
        Duplicate(tmpLocal, ONE, 1);
        PipeBarrier<PIPE_V>();
        Div(sqxLocal, tmpLocal, sqxLocal, 1);
        PipeBarrier<PIPE_V>();

        // copyout rstd
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<float>(rstdGm, sqxLocal, 1);
        event_t eventMTE3V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3V2);
#endif
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float rstdValue = sqxLocal.GetValue(0);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Muls(xFp32Local, xFp32Local, rstdValue, numCol);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V2);
#endif
        Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        Mul(xFp32Local, xFp32Local, sqxLocal, numCol);
        PipeBarrier<PIPE_V>();
        Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<T>(y2Gm, x1Local, numCol);
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<float>(y1Gm, xFp32Local, numCol);
    }

private:
    TPipe *Ppipe = nullptr;

    TBuf<TPosition::VECCALC> unitBuf;
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
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    int32_t blockIdx_;
    uint32_t rowWork = 1;
};
#endif  // ADD_RMS_NORM_CAST_SINGLE_N_H_