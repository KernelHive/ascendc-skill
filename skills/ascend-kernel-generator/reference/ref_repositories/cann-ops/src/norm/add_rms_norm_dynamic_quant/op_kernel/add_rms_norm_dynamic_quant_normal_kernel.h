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
 * \file add_rms_norm_dynamic_quant_normal_kernel.h
 * \brief
 */

#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_NORMAL_KERNEL_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_NORMAL_KERNEL_H_

#include "add_rms_norm_dynamic_quant_base.h"

template <typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddRmsNormDynamicQuantNormal : public KernelAddRmsNormDynamicQuantBase<T, TILING_KEY, BUFFER_NUM> {
public:
    __aicore__ inline KernelAddRmsNormDynamicQuantNormal(TPipe *pipe)
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
        this->numRowsAligned = (this->rowStep + ELEM_PER_BLK_FP32 - 1) / ELEM_PER_BLK_FP32 * ELEM_PER_BLK_FP32;
        this->ubAligned = (this->numLastDimAligned - this->numLastDim) >= ELEM_PER_BLK_FP16;
        /*
          UB = 3 * this->rowStep * alignedCol * sizeof(T)
              + 2 * this->rowStep * alignedCol * sizeof(float)
              + Count(gamma,beta,bias) * alignedCol * sizeof(T)
              + 512Bytes(256 + reduceOut)
        */
        Ppipe->InitBuffer(inRowsQue, BUFFER_NUM, 2 * this->rowStep * this->numLastDimAligned * sizeof(T));  // 2 * D * 2
        Ppipe->InitBuffer(outRowsQue, BUFFER_NUM, 2 * this->rowStep * this->numLastDimAligned * sizeof(T));  // D * 2
        Ppipe->InitBuffer(xBufFp32, this->rowStep * this->numLastDimAligned * sizeof(float));                // D * 4
        Ppipe->InitBuffer(yBufFp32, this->rowStep * this->numLastDimAligned * sizeof(float));                // D * 4
        Ppipe->InitBuffer(weightBuf01, this->numLastDimAligned * sizeof(T));                                 // D * 2
        Ppipe->InitBuffer(weightBuf02, this->numLastDimAligned * sizeof(T));                                 // D * 2
        Ppipe->InitBuffer(weightBuf03, this->numLastDimAligned * sizeof(T));                                 // D * 2
        // 2 dynamic quant operator required 2 scale buffer.
        Ppipe->InitBuffer(scalesBuf, 2 * this->numRowsAligned * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t rowMoveCnt = CEIL_DIV(this->rowWork, this->rowStep);
        CopyInWeights();

        LocalTensor<T> gammaLocal = weightBuf01.template Get<T>();

        int32_t gmOffset = 0;
        int32_t gmOffsetScale = 0;
        int32_t elementCount = this->numLastDimAligned * this->rowStep;

        for (int32_t rowIdx = 0; rowIdx < rowMoveCnt - 1; ++rowIdx) {
            CopyInX1X2(gmOffset, this->rowStep, elementCount);
            AddX1X2(gmOffset, elementCount);
            CopyOutX(gmOffset, this->rowStep, elementCount);
            ComputeRmsNorm(this->rowStep, elementCount, gammaLocal);
            ComputeDynamicQuant(this->rowStep, elementCount);
            CopyOut(gmOffset, gmOffsetScale, this->rowStep);
            gmOffset += this->rowStep * this->numLastDim;
            gmOffsetScale += this->rowStep;
        }
        {
            elementCount = this->numLastDimAligned * this->rowTail_;
            int32_t rowIdx = rowMoveCnt - 1;
            CopyInX1X2(gmOffset, this->rowTail_, elementCount);
            AddX1X2(gmOffset, elementCount);
            CopyOutX(gmOffset, this->rowTail_, elementCount);
            ComputeRmsNorm(this->rowTail_, elementCount, gammaLocal);
            ComputeDynamicQuant(this->rowTail_, elementCount);
            CopyOut(gmOffset, gmOffsetScale, this->rowTail_);
        }
    }

private:
    __aicore__ inline void CopyInX1X2(int32_t gmOffset, int32_t rowCount, int32_t elementCount)
    {
        LocalTensor<T> x1x2LocalIn = inRowsQue.template AllocTensor<T>();
        DataCopyEx(x1x2LocalIn[0], this->x2Gm[gmOffset], this->numLastDim, rowCount, this->ubAligned);
        DataCopyEx(x1x2LocalIn[elementCount], this->x1Gm[gmOffset], this->numLastDim, rowCount, this->ubAligned);
        inRowsQue.EnQue(x1x2LocalIn);
    }

    __aicore__ inline void AddX1X2(int32_t gmOffset, int32_t elementCount)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

        LocalTensor<T> x1x2Local = inRowsQue.template DeQue<T>();
        auto x1Local = x1x2Local[elementCount];
        auto x2Local = x1x2Local[0];
        // never have fp32 input here. All fp16/bf16 should cast to fp32 before Add
        Cast(xLocalFp32, x1Local, RoundMode::CAST_NONE, elementCount);
        Cast(yLocalFp32, x2Local, RoundMode::CAST_NONE, elementCount);
        inRowsQue.FreeTensor(x1x2Local);
        PipeBarrier<PIPE_V>();
        Add(xLocalFp32, xLocalFp32, yLocalFp32, elementCount);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOutX(int32_t gmOffset, int32_t rowCount, int32_t elementCount)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<T> xOut = outRowsQue.template AllocTensor<T>();
        if constexpr (IsSame<T, half>::value) {
            Cast(xOut, xLocalFp32, RoundMode::CAST_NONE, elementCount);
        } else {  // BF16
            Cast(xOut, xLocalFp32, RoundMode::CAST_RINT, elementCount);
        }
        PipeBarrier<PIPE_V>();
        outRowsQue.EnQue(xOut);
        LocalTensor<T> x = outRowsQue.template DeQue<T>();
        DataCopyEx(this->xGm[gmOffset], x, this->numLastDim, rowCount, this->ubAligned);
        outRowsQue.FreeTensor(x);
    }

    __aicore__ inline void CopyOutY(int32_t gmOffset, int32_t rowCount, int32_t elementCount)
    {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<float> yLocal = xBufFp32.Get<float>();
        LocalTensor<T> yOut = yBufFp32.Get<T>();
        PipeBarrier<PIPE_ALL>();
        if constexpr (IsSame<T, half>::value) {
            Cast(yOut, yLocal, RoundMode::CAST_NONE, elementCount);
        } else {  // BF16
            Cast(yOut, yLocal, RoundMode::CAST_RINT, elementCount);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyEx(this->xGm[gmOffset], yOut, this->numLastDim, rowCount, this->ubAligned);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInWeights()
    {
        LocalTensor<T> gammaLocal = weightBuf01.template Get<T>();
        DataCopyEx(gammaLocal, this->gammaGm, this->numLastDim);
        if (this->smooth1Exist) {
            LocalTensor<T> smooth1Local = weightBuf02.template Get<T>();
            DataCopyEx(smooth1Local, this->smooth1Gm, this->numLastDim);
        }
        if (this->smooth2Exist) {
            LocalTensor<T> smooth2Local = weightBuf03.template Get<T>();
            DataCopyEx(smooth2Local, this->smooth2Gm, this->numLastDim);
        }
    }

    __aicore__ inline void ComputeRmsNorm(int32_t nums, int32_t elementCount, LocalTensor<T> &gammaLocal)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();  // xLocalFp32 <-- x1 + x2
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

        Mul(yLocalFp32, xLocalFp32, xLocalFp32, elementCount);  // yLocalFp32 <- x ** 2
        PipeBarrier<PIPE_V>();

        // reduce#1 for mean
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto roundOffset = rid * this->numLastDimAligned;
            float squareSumTemp =
                ReduceSumHalfInterval(yLocalFp32[roundOffset], this->numLastDim);  // aveLocalTemp <-- E(x**2)
            float rstdLocalTemp = 1 / sqrt(squareSumTemp * this->aveNum + this->eps);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            Muls(xLocalFp32[roundOffset],
                xLocalFp32[roundOffset],
                rstdLocalTemp,
                this->numLastDim);  // xLocalFp32 <- x * rstd
        }
        PipeBarrier<PIPE_V>();

        Cast(yLocalFp32, gammaLocal, RoundMode::CAST_NONE, this->numLastDim);  // yLocalFp32 <- gamma
        PipeBarrier<PIPE_V>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto roundOffset = rid * this->numLastDimAligned;
            Mul(xLocalFp32[roundOffset],
                xLocalFp32[roundOffset],
                yLocalFp32,
                this->numLastDim);  // xLocalFp32 <- x * rstd * gamma
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeDynamicQuant(int32_t nums, int32_t elementCount)
    {
        if (!this->smooth1Exist) {
            ComputeZeroDynamicQuant(nums, elementCount);
        } else if (!this->smooth2Exist) {
            ComputeSoleDynamicQuant(nums, elementCount);
        } else {
            ComputeDualDynamicQuant(nums, elementCount);
        }
    }

    __aicore__ inline void ComputeDualDynamicQuant(int32_t nums, int32_t elementCount)
    {
        LocalTensor<float> scaleLocal = scalesBuf.Get<float>();
        LocalTensor<float> scale1Local = scaleLocal[0];
        LocalTensor<float> scale2Local = scaleLocal[this->numRowsAligned];
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();  // xLocalFp32 <-- y
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<float> zLocalFp32 = outRowsQue.template AllocTensor<float>();
        LocalTensor<int8_t> outQuant01 = zLocalFp32.ReinterpretCast<int8_t>();
        LocalTensor<int8_t> outQuant02 = outQuant01[elementCount];
        LocalTensor<T> smooth1Local = weightBuf02.Get<T>();
        LocalTensor<T> smooth2Local = weightBuf03.Get<T>();

        // compute smooth1
        LocalTensor<float> smooth1Fp32 = yLocalFp32[(nums - 1) * this->numLastDimAligned];
        LocalTensor<float> smooth2Fp32 = zLocalFp32[(nums - 1) * this->numLastDimAligned];
        Cast(smooth1Fp32, smooth1Local, RoundMode::CAST_NONE, this->numLastDim);
        Cast(smooth2Fp32, smooth2Local, RoundMode::CAST_NONE, this->numLastDim);
        PipeBarrier<PIPE_V>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            Mul(yLocalFp32[rid * this->numLastDimAligned],
                xLocalFp32[rid * this->numLastDimAligned],
                smooth1Fp32,
                this->numLastDim);  // yLocalFp32 <-- y * smooth1
            Mul(zLocalFp32[rid * this->numLastDimAligned],
                xLocalFp32[rid * this->numLastDimAligned],
                smooth2Fp32,
                this->numLastDim);  // xLocalFp32 <-- y * smooth2
        }
        PipeBarrier<PIPE_V>();

        ScaleTensor(yLocalFp32, xLocalFp32, scale1Local, elementCount, nums);
        PipeBarrier<PIPE_V>();
        ScaleTensor(zLocalFp32, xLocalFp32, scale2Local, elementCount, nums);
        PipeBarrier<PIPE_V>();

        Cast(yLocalFp32.ReinterpretCast<int32_t>(), yLocalFp32, RoundMode::CAST_RINT, elementCount);
        Cast(xLocalFp32.ReinterpretCast<int32_t>(), zLocalFp32, RoundMode::CAST_RINT, elementCount);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(yLocalFp32.ReinterpretCast<half>(),
            yLocalFp32.ReinterpretCast<int32_t>(),
            RoundMode::CAST_NONE,
            elementCount);
        Cast(xLocalFp32.ReinterpretCast<half>(),
            xLocalFp32.ReinterpretCast<int32_t>(),
            RoundMode::CAST_NONE,
            elementCount);
        PipeBarrier<PIPE_V>();
        Cast(outQuant01, yLocalFp32.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, elementCount);
        Cast(outQuant02, xLocalFp32.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, elementCount);
        PipeBarrier<PIPE_V>();
        outRowsQue.EnQue(zLocalFp32);
    }

    __aicore__ inline void CopyOut(int32_t gmOffset, int32_t gmOffsetScale, int32_t rowCount)
    {
        LocalTensor<int8_t> outY12 = outRowsQue.template DeQue<int8_t>();
        LocalTensor<float> scaleLocal = scalesBuf.Get<float>();
        LocalTensor<int8_t> outQuant01 = outY12[0];
        LocalTensor<int8_t> outQuant02 = outY12[rowCount * this->numLastDimAligned];
        LocalTensor<float> scale1Local = scaleLocal[0];
        LocalTensor<float> scale2Local = scaleLocal[this->numRowsAligned];

        DataCopyEx(this->y1Gm[gmOffset], outQuant01, this->numLastDim, rowCount);
        DataCopyEx(this->outScale1Gm[gmOffsetScale], scale1Local, rowCount);
        if (this->smooth2Exist) {
            DataCopyEx(this->y2Gm[gmOffset], outQuant02, this->numLastDim, rowCount);
            DataCopyEx(this->outScale2Gm[gmOffsetScale], scale2Local, rowCount);
        }
        outRowsQue.FreeTensor(outY12);
    }

    __aicore__ inline void ScaleTensor(LocalTensor<float> &srcTensor, LocalTensor<float> &tmpTensor,
        LocalTensor<float> &scaleTensor, int32_t size, int32_t nums)
    {
        float maxTemp;
        float scaleTemp;
        event_t eventVS;
        event_t eventSV;
        Abs(tmpTensor, srcTensor, size);  // tmpLocal <-- |y * smooth1|
        PipeBarrier<PIPE_V>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            ReduceMaxInplace(tmpTensor[rid * this->numLastDimAligned], this->numLastDim);
            eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            maxTemp = tmpTensor[rid * this->numLastDimAligned].GetValue(0);  // Reduce
            scaleTemp = DYNAMIC_QUANT_DIVIDEND / maxTemp;
            scaleTensor.SetValue(rid, 1 / scaleTemp);
            eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            auto srcSlice = srcTensor[rid * this->numLastDimAligned];
            Muls(srcSlice, srcSlice, scaleTemp, this->numLastDim);
        }
    }

    __aicore__ inline void ComputeZeroDynamicQuant(int32_t nums, int32_t elementCount)
    {
        LocalTensor<float> scaleLocal = scalesBuf.Get<float>();
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();  // xLocalFp32 <-- RmsNorm(x)
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        ScaleTensor(xLocalFp32, yLocalFp32, scaleLocal, elementCount, nums);
        PipeBarrier<PIPE_V>();
        LocalTensor<int8_t> yLocal = outRowsQue.template AllocTensor<int8_t>();

        RoundFloat2Int8(yLocal, xLocalFp32, elementCount);
        PipeBarrier<PIPE_V>();
        outRowsQue.EnQue(yLocal);
    }

    __aicore__ inline void ComputeSoleDynamicQuant(int32_t nums, int32_t elementCount)
    {
        LocalTensor<float> scaleLocal = scalesBuf.Get<float>();
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();  // xLocalFp32 <-- y
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<T> smoothLocal = weightBuf02.Get<T>();

        // compute smooth1
        auto smoothFp32 = yLocalFp32[(nums - 1) * this->numLastDimAligned];
        Cast(smoothFp32, smoothLocal, RoundMode::CAST_NONE, this->numLastDim);
        PipeBarrier<PIPE_V>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            Mul(yLocalFp32[rid * this->numLastDimAligned],
                xLocalFp32[rid * this->numLastDimAligned],
                smoothFp32,
                this->numLastDim);
        }
        PipeBarrier<PIPE_V>();
        ScaleTensor(yLocalFp32, xLocalFp32, scaleLocal, elementCount, nums);
        PipeBarrier<PIPE_V>();
        LocalTensor<int8_t> yLocal = outRowsQue.template AllocTensor<int8_t>();
        RoundFloat2Int8(yLocal, yLocalFp32, elementCount);
        outRowsQue.EnQue(yLocal);
    }

private:
    TPipe *Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inRowsQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outRowsQue;

    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;

    TBuf<TPosition::VECCALC> weightBuf01;
    TBuf<TPosition::VECCALC> weightBuf02;
    TBuf<TPosition::VECCALC> weightBuf03;
    TBuf<TPosition::VECCALC> scalesBuf;

    uint32_t numRowsAligned;
    bool ubAligned;
};

#endif  // ADD_RMS_NORM_DYNAMIC_QUANT_NORMAL_KERNEL_H_
