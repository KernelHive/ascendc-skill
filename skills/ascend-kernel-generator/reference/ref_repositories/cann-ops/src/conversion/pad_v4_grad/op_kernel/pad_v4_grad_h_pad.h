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
 * \file pad_v4_grad_h_pad.h
 * \brief
 */
#ifndef _PAD_V4_GRAD_H_PAD_H_
#define _PAD_V4_GRAD_H_PAD_H_

#include "pad_v4_grad_base.h"

template <typename T>
class PadV4GradPadH : public PadV4GradBase<T> {
   public:
    __aicore__ inline PadV4GradPadH(){};
    __aicore__ inline void InitBuffer(TPipe *inputPipe);
    __aicore__ inline void CopyGm2UB(const int64_t offset1, const int64_t offset2, const int64_t copyCount);
    __aicore__ inline void ComputeHGrad(const int32_t calCount);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(const int64_t copyCount, const int64_t offset);
    __aicore__ inline void Compute(const int64_t copyCount);
    __aicore__ inline void CopyOut(const int64_t copyCount, const int64_t offset);

   private:
    TPipe *pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yOutQueue;
    TBuf<TPosition::VECCALC> floatCastResBuf;

    event_t eventId0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    LocalTensor<float> floatTenosr;
};

// init used buffer
template <typename T>
__aicore__ inline void PadV4GradPadH<T>::InitBuffer(TPipe *inputPipe) {
    pipe = inputPipe;
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        pipe->InitBuffer(xInQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T) * 2);
        pipe->InitBuffer(yOutQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T) * 2);
        pipe->InitBuffer(floatCastResBuf, this->ubFactorElement * sizeof(float) * 2);
    } else {
        pipe->InitBuffer(xInQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T) * 2);
        pipe->InitBuffer(yOutQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T) * 2);
    }
}

template <typename T>
__aicore__ inline void PadV4GradPadH<T>::CopyGm2UB(const int64_t offset1, const int64_t offset2,
                                                   const int64_t copyCount) {
    LocalTensor<T> dataLocal = xInQueue.AllocTensor<T>();
    int32_t alignCopyCount = this->CeilAlign(copyCount, this->perBlockCount);
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {true, 0, (uint8_t)(alignCopyCount - copyCount), (T)0};
    DataCopyPad(dataLocal[0], this->mGmX[offset1], copyParams, padParams);
    DataCopyPad(dataLocal[this->ubFactorElement], this->mGmX[offset2], copyParams, padParams);
    xInQueue.EnQue(dataLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadH<T>::CopyIn(const int64_t copyCount, const int64_t offset) {
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    int32_t alignCopyCount = this->CeilAlign(copyCount, this->perBlockCount);
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {true, 0, (uint8_t)(alignCopyCount - copyCount), (T)0};
    DataCopyPad(xLocal, this->mGmX[offset], copyParams, padParams);
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadH<T>::Compute(const int64_t copyCount) {
    int32_t alignCopyCount = this->CeilAlign(copyCount, this->perBlockCount);
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    DataCopy(yLocal, xLocal, alignCopyCount);
    yOutQueue.EnQue(yLocal);
    xInQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadH<T>::CopyOut(const int64_t copyCount, const int64_t offset) {
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(this->mGmY[offset], yLocal, copyParams);
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadH<T>::ComputeHGrad(const int32_t calCount) {
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        Cast(floatTenosr, xLocal, RoundMode::CAST_NONE, 2 * this->ubFactorElement);
        Add(floatTenosr, floatTenosr[0], floatTenosr[this->ubFactorElement], calCount);
        Cast(yLocal, floatTenosr, RoundMode::CAST_RINT, calCount);
    } else {
        Add(yLocal, xLocal[0], xLocal[this->ubFactorElement], calCount);
    }
    yOutQueue.EnQue(yLocal);
    xInQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadH<T>::Process() {
    int64_t gmXOffset = 0;
    int64_t gmXOffset1;
    int64_t gmXOffset2;
    int64_t gmXOffset3;
    int64_t gmXOffset4;
    int64_t gmXOffset5;
    int64_t gmXOffset6;
    int64_t gmXOffset7;
    int64_t gmYOffset1;
    int64_t gmYOffset2;
    int64_t gmYOffset3;
    int64_t gmYOffset4;
    int64_t gmYOffset5;
    int64_t calCount = 0;
    int64_t copyCount = 0;
    int64_t firstAndLastRowCopyCount = 0;
    int64_t midDataCount = this->width * (this->height - (1 + 2 * this->hPad2) - (2 * this->hPad1 + 1));
    uint32_t copyTimesOneRow1 = this->CeilDiv(this->width, this->ubFactorElement);
    uint32_t copyTimesOneRow2 = this->CeilDiv(this->width, 2 * this->ubFactorElement);
    uint32_t copyTimesMidData = this->CeilDiv(midDataCount, 2 * this->ubFactorElement);
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        floatTenosr = floatCastResBuf.Get<float>();
    }
    for (size_t loop = 0; loop < this->loopNC; loop++) {
        calCount = this->ubFactorElement;
        copyCount = this->ubFactorElement * 2;
        firstAndLastRowCopyCount = this->ubFactorElement * 2;

        for (size_t time = 0; time < copyTimesOneRow1; time++) {
            if (time == copyTimesOneRow1 - 1) {
                calCount = this->width - (copyTimesOneRow1 - 1) * this->ubFactorElement;
            }
            for (size_t i = 0; i < this->hPad1; i++) {
                // 搬两行
                gmXOffset1 = i * this->width + time * this->ubFactorElement + loop * this->batchStride +
                             this->ncOffset * this->batchStride;
                gmXOffset2 = (2 * this->hPad1 - i) * this->width + time * this->ubFactorElement +
                             loop * this->batchStride + this->ncOffset * this->batchStride;
                gmYOffset1 = (this->hPad1 - i) * this->width + time * this->ubFactorElement +
                             loop * this->outBatchStride + this->ncOffset * this->outBatchStride;

                CopyGm2UB(gmXOffset1, gmXOffset2, calCount);
                ComputeHGrad(calCount);
                CopyOut(calCount, gmYOffset1);
            }
            for (size_t j = 0; j < this->hPad2; j++) {
                // 搬两行
                gmXOffset3 = ((this->height - 2 * this->hPad2 - 1) + j) * this->width + time * this->ubFactorElement +
                             loop * this->batchStride + this->ncOffset * this->batchStride;
                gmXOffset4 = (this->height - 1 - j) * this->width + time * this->ubFactorElement +
                             loop * this->batchStride + this->ncOffset * this->batchStride;
                gmYOffset2 = (this->outHeight - this->hPad2 - 1 + j) * this->width + time * this->ubFactorElement +
                             loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
                CopyGm2UB(gmXOffset3, gmXOffset4, calCount);
                ComputeHGrad(calCount);
                CopyOut(calCount, gmYOffset2);
            }
        }

        copyCount = this->ubFactorElement * 2;
        for (size_t time = 0; time < copyTimesMidData; time++) {
            if (time == copyTimesMidData - 1) {
                copyCount = midDataCount - (copyTimesMidData - 1) * this->ubFactorElement * 2;
            }
            gmXOffset5 = (2 * this->hPad1 + 1) * this->width + time * this->ubFactorElement * 2 +
                         loop * this->batchStride + this->ncOffset * this->batchStride;
            gmYOffset3 = (this->hPad1 + 1) * this->width + time * this->ubFactorElement * 2 +
                         loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
            CopyIn(copyCount, gmXOffset5);
            Compute(copyCount);
            CopyOut(copyCount, gmYOffset3);
        }
        for (size_t time = 0; time < copyTimesOneRow2; time++) {
            if (time == copyTimesOneRow2 - 1) {
                firstAndLastRowCopyCount = this->width - (copyTimesOneRow2 - 1) * this->ubFactorElement * 2;
            }
            gmXOffset6 = this->hPad1 * this->width + time * this->ubFactorElement * 2 + loop * this->batchStride +
                         this->ncOffset * this->batchStride;
            gmYOffset4 =
                time * this->ubFactorElement * 2 + loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
            gmXOffset7 = (this->height - this->hPad2 - 1) * this->width + time * this->ubFactorElement * 2 +
                         loop * this->batchStride + this->ncOffset * this->batchStride;
            gmYOffset5 = (this->outHeight - 1) * this->width + time * this->ubFactorElement * 2 +
                         loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
            // copy 第一行和最后一行
            CopyIn(firstAndLastRowCopyCount, gmXOffset6);
            Compute(firstAndLastRowCopyCount);
            CopyOut(firstAndLastRowCopyCount, gmYOffset4);
            CopyIn(firstAndLastRowCopyCount, gmXOffset7);
            Compute(firstAndLastRowCopyCount);
            CopyOut(firstAndLastRowCopyCount, gmYOffset5);
        }
    }
}
#endif  // _PAD_V4_GRAD_H_PAD_H_