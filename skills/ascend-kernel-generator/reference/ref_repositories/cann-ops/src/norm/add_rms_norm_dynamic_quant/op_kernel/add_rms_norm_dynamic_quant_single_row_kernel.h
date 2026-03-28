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
 * \file add_rms_norm_dynamic_quant_single_row_kernel.h
 * \brief
 */

#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_SINGLE_ROW_KERNEL_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_SINGLE_ROW_KERNEL_H_

#include "add_rms_norm_dynamic_quant_base.h"

template <typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddRmsNormDynamicQuantSingleRow : public KernelAddRmsNormDynamicQuantBase<T, TILING_KEY, BUFFER_NUM> {
public:
    __aicore__ inline KernelAddRmsNormDynamicQuantSingleRow(TPipe *pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooth1, GM_ADDR smooth2, GM_ADDR y1,
        GM_ADDR y2, GM_ADDR x, GM_ADDR outScale1, GM_ADDR outScale2, GM_ADDR workspace,
        const AddRmsNormDynamicQuantTilingData *tiling)
    {
        this->InitBaseParams(tiling);
        this->InitInGlobalTensors(x1, x2, gamma, smooth1, smooth2);
        this->InitOutGlobalTensors(y1, y2, x, outScale1, outScale2);

        /*
          UB = 3 * alignedCol * sizeof(T)
              + 2 * alignedCol * sizeof(float)
              + Count(bias) * alignedCol * sizeof(T)
              + 512Btyes(256 + reduceOut)
        */
        Ppipe->InitBuffer(inRowsQue, BUFFER_NUM, 2 * this->numLastDimAligned * sizeof(T));  // 2 * D * 2
        Ppipe->InitBuffer(yQue, BUFFER_NUM, this->numLastDimAligned * sizeof(T));           // D * 2

        Ppipe->InitBuffer(xBufFp32, this->numLastDimAligned * sizeof(float));  // D * 4
        Ppipe->InitBuffer(yBufFp32, this->numLastDimAligned * sizeof(float));  // D * 4
        Ppipe->InitBuffer(smoothBuf, this->numLastDimAligned * sizeof(T));     // D * 2

        // 2 dynamic quant operator required 2 scale buffer.
        Ppipe->InitBuffer(scalesQue, BUFFER_NUM, 2 * ROW_FACTOR * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->smooth1Exist) {
            LocalTensor<T> smooth1Local = smoothBuf.template Get<T>();
            DataCopyEx(smooth1Local, this->smooth1Gm, this->numLastDim);
        }

        int32_t outLoopCount = this->rowWork / ROW_FACTOR;
        int32_t outLoopTail = this->rowWork % ROW_FACTOR;
        uint32_t gmOffset = 0;
        uint32_t gmOffsetReduce = 0;

        LocalTensor<float> scalesLocalOut;

        for (int32_t loopIdx = 0; loopIdx < outLoopCount; ++loopIdx) {
            scalesLocalOut = scalesQue.template AllocTensor<float>();
            for (int32_t innerIdx = 0; innerIdx < ROW_FACTOR; ++innerIdx) {
                CopyInX1X2(gmOffset);
                AddSingleRow(gmOffset);
                CopyInGamma();
                ComputeRmsNorm(gmOffset);
                CopyInSmooth();
                ComputeDynamicQuant(innerIdx, scalesLocalOut, gmOffset);
                CopyOut(gmOffset);
                gmOffset += this->numLastDim;
            }
            scalesQue.EnQue(scalesLocalOut);
            CopyOutScale(gmOffsetReduce, ROW_FACTOR);
            gmOffsetReduce += ROW_FACTOR;
        }
        {
            scalesLocalOut = scalesQue.template AllocTensor<float>();
            for (int32_t innerIdx = 0; innerIdx < outLoopTail; ++innerIdx) {
                CopyInX1X2(gmOffset);
                AddSingleRow(gmOffset);
                CopyInGamma();
                ComputeRmsNorm(gmOffset);
                CopyInSmooth();
                ComputeDynamicQuant(innerIdx, scalesLocalOut, gmOffset);
                CopyOut(gmOffset);
                gmOffset += this->numLastDim;
            }
            scalesQue.EnQue(scalesLocalOut);
            CopyOutScale(gmOffsetReduce, outLoopTail);
        }
    }

private:
    __aicore__ inline void AddSingleRow(int32_t gmOffset)
    {
        auto x1x2Local = inRowsQue.template DeQue<T>();
        auto x1Local = x1x2Local[0];
        auto x2Local = x1x2Local[this->numLastDimAligned];

        auto xBufLocal = xBufFp32.Get<float>();
        auto yBufLocal = yBufFp32.Get<float>();

        // never have fp32 input here. All fp16/bf16 should cast to fp32 before Add
        Cast(xBufLocal, x1Local, RoundMode::CAST_NONE, this->numLastDim);
        Cast(yBufLocal, x2Local, RoundMode::CAST_NONE, this->numLastDim);
        inRowsQue.FreeTensor(x1x2Local);
        PipeBarrier<PIPE_V>();
        Add(xBufLocal, yBufLocal, xBufLocal, this->numLastDim);
        PipeBarrier<PIPE_V>();
        auto xLocal = yQue.template AllocTensor<T>();
        Cast(xLocal, xBufLocal, RoundMode::CAST_RINT, this->numLastDim);
        yQue.template EnQue<T>(xLocal);

        PipeBarrier<PIPE_V>();
        auto x = yQue.template DeQue<T>();
        DataCopyEx(this->xGm[gmOffset], x, this->numLastDim);
        yQue.FreeTensor(x);
    }

    __aicore__ inline void ComputeRmsNorm(int32_t gmOffset)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<T> yLocalB16 = yBufFp32.Get<T>();

        Mul(yLocalFp32, xLocalFp32, xLocalFp32, this->numLastDim);  // yLocalFp32 <- x ** 2
        PipeBarrier<PIPE_V>();

        float squareSumTemp = ReduceSumHalfInterval(yLocalFp32, this->numLastDim);
        float rstdLocalTemp = 1 / sqrt(squareSumTemp * this->aveNum + this->eps);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Muls(xLocalFp32, xLocalFp32, rstdLocalTemp, this->numLastDim);  // xLocalFp32 <- x * rstd
        PipeBarrier<PIPE_V>();
        LocalTensor<T> gammaLocal = inRowsQue.template DeQue<T>();

        Cast(yLocalFp32, gammaLocal, RoundMode::CAST_NONE, this->numLastDim);  // yLocalB16 <- Cast(gamma)
        inRowsQue.FreeTensor(gammaLocal);
        Mul(xLocalFp32, xLocalFp32, yLocalFp32, this->numLastDim);  // xLocalFp32 <- x * rstd * gamma
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeDynamicQuant(int32_t idx, LocalTensor<float> &scalesLocalOut, int32_t gmOffset)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

        LocalTensor<int8_t> yLocal = yQue.template AllocTensor<int8_t>();

        if (!this->smooth1Exist) {
            ScaleTensor(xLocalFp32, yLocalFp32, scalesLocalOut, idx);
            PipeBarrier<PIPE_V>();
            RoundFloat2Int8(yLocal, xLocalFp32, this->numLastDim);
        } else if (!this->smooth2Exist) {
            LocalTensor<T> smooth1Local = smoothBuf.template Get<T>();
            Cast(yLocalFp32, smooth1Local, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Mul(xLocalFp32, xLocalFp32, yLocalFp32, this->numLastDim);
            PipeBarrier<PIPE_V>();
            ScaleTensor(xLocalFp32, yLocalFp32, scalesLocalOut, idx);
            PipeBarrier<PIPE_V>();
            RoundFloat2Int8(yLocal, xLocalFp32, this->numLastDim);
        } else {
            LocalTensor<T> smooth1Local = smoothBuf.template Get<T>();
            LocalTensor<T> smooth2Local = inRowsQue.template DeQue<T>();
            LocalTensor<float> tmpTensor = smooth2Local.template ReinterpretCast<float>();
            auto y1Local = yLocal[0];
            auto y2Local = yLocal[this->numLastDimAligned];

            Cast(yLocalFp32, smooth2Local, RoundMode::CAST_NONE, this->numLastDim);  // yLocalFp32 <-- smooth2
            PipeBarrier<PIPE_V>();
            Mul(yLocalFp32, xLocalFp32, yLocalFp32, this->numLastDim);  // yLocalFp32 <-- y * yLocalFp32
            PipeBarrier<PIPE_V>();
            ScaleTensor(yLocalFp32,
                tmpTensor,
                scalesLocalOut,
                idx + ROW_FACTOR);  // yLocalFp32 <-- yLocalFp32 / max(abs(yLocalFp32))
            PipeBarrier<PIPE_V>();
            inRowsQue.FreeTensor(tmpTensor);
            RoundFloat2Int8(y2Local, yLocalFp32, this->numLastDim);

            Cast(yLocalFp32, smooth1Local, RoundMode::CAST_NONE, this->numLastDim);  // yLocalFp32 <-- smooth1
            PipeBarrier<PIPE_V>();
            Mul(yLocalFp32, xLocalFp32, yLocalFp32, this->numLastDim);  // yLocalFp32 <-- y * smooth1
            PipeBarrier<PIPE_V>();
            ScaleTensor(
                yLocalFp32, xLocalFp32, scalesLocalOut, idx);  // yLocalFp32 <-- yLocalFp32 / max(abs(yLocalFp32))
            PipeBarrier<PIPE_V>();
            RoundFloat2Int8(y1Local, yLocalFp32, this->numLastDim);
        }
        PipeBarrier<PIPE_V>();
        yQue.EnQue(yLocal);
    }

    // srcTensor <- srcTensor / max(abs(srcTensor))
    __aicore__ inline void ScaleTensor(
        LocalTensor<float> &srcTensor, LocalTensor<float> &tmpTensor, LocalTensor<float> &scaleTensor, int32_t idx)
    {
        Abs(tmpTensor, srcTensor, this->numLastDim);  // tmpLocal <-- |y * smooth|
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(tmpTensor, this->numLastDim);
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float maxTemp = tmpTensor.GetValue(0);
        float scaleTemp = DYNAMIC_QUANT_DIVIDEND / maxTemp;
        scaleTensor.SetValue(idx, 1 / scaleTemp);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Muls(srcTensor, srcTensor, scaleTemp, this->numLastDim);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOut(int32_t gmOffset)
    {
        LocalTensor<int8_t> res12 = yQue.template DeQue<int8_t>();
        auto res1 = res12[0];
        auto res2 = res12[this->numLastDimAligned];
        DataCopyEx(this->y1Gm[gmOffset], res1, this->numLastDim);
        if (this->smooth2Exist) {
            DataCopyEx(this->y2Gm[gmOffset], res2, this->numLastDim);
        }
        yQue.FreeTensor(res12);
    }

    __aicore__ inline void CopyOutScale(int32_t gmOffset, int32_t copyInNums)
    {
        LocalTensor<float> outScalesLocal = scalesQue.template DeQue<float>();
        LocalTensor<float> outScales1Local = outScalesLocal[0];
        LocalTensor<float> outScales2Local = outScalesLocal[ROW_FACTOR];
        DataCopyEx(this->outScale1Gm[gmOffset], outScales1Local, copyInNums);
        if (this->smooth2Exist) {
            DataCopyEx(this->outScale2Gm[gmOffset], outScales2Local, copyInNums);
        }
        scalesQue.FreeTensor(outScalesLocal);
    }

    __aicore__ inline void CopyInX1X2(int32_t gmOffset)
    {
        LocalTensor<T> x1x2LocalIn = inRowsQue.template AllocTensor<T>();
        DataCopyEx(x1x2LocalIn[0], this->x1Gm[gmOffset], this->numLastDim);
        DataCopyEx(x1x2LocalIn[this->numLastDimAligned], this->x2Gm[gmOffset], this->numLastDim);
        inRowsQue.EnQue(x1x2LocalIn);
    }

    __aicore__ inline void CopyInSmooth()
    {
        if (this->smooth2Exist) {
            LocalTensor<T> smoothCopyIn = inRowsQue.template AllocTensor<T>();
            DataCopyEx(smoothCopyIn[0], this->smooth2Gm, this->numLastDim);
            inRowsQue.EnQue(smoothCopyIn);
        }
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T> gammaCopyIn = inRowsQue.template AllocTensor<T>();
        DataCopyEx(gammaCopyIn[0], this->gammaGm, this->numLastDim);
        inRowsQue.EnQue(gammaCopyIn);
    }

private:
    TPipe *Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inRowsQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> scalesQue;

    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;

    TBuf<TPosition::VECCALC> smoothBuf;
};

#endif  // ADD_RMS_NORM_DYNAMIC_QUANT_SINGLE_ROW_KERNEL_H_
