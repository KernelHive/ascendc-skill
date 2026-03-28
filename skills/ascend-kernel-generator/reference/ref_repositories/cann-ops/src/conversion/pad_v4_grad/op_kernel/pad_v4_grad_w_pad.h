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
 * \file pad_v4_grad_w_pad.h
 * \brief
 */
#ifndef _PAD_V4_GRAD_W_PAD_H_
#define _PAD_V4_GRAD_W_PAD_H_

#include "pad_v4_grad_base.h"

template <typename T>
class PadV4GradPadW : public PadV4GradBase<T> {
   public:
    __aicore__ inline PadV4GradPadW(){};
    __aicore__ inline void InitBuffer(TPipe* inputPipe);
    __aicore__ inline void CopyGm2UB(TQue<QuePosition::VECIN, BUFFER_NUM>& inQue, const int64_t offset1,
                                     const int64_t offset2, const int64_t copyCount);
    __aicore__ inline void ComputeWGrad(TQue<QuePosition::VECIN, BUFFER_NUM>& inQue,
                                        TQue<QuePosition::VECOUT, BUFFER_NUM>& outQue, const int32_t calCount);
    __aicore__ inline void Bf16ComputeWGrad(TQue<QuePosition::VECIN, BUFFER_NUM>& inQue,
                                            TQue<QuePosition::VECOUT, BUFFER_NUM>& outQue, const int32_t calCount,
                                            const int32_t castCount, LocalTensor<float> castTensor);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(const int64_t copyCount, const int64_t offset);
    __aicore__ inline void Compute(TQue<QuePosition::VECIN, BUFFER_NUM>& inQue,
                                   TQue<QuePosition::VECOUT, BUFFER_NUM>& outQue, const int64_t copyCount);
    __aicore__ inline void CopyOut(const int64_t copyCount, const int64_t offset);
    __aicore__ inline void CopyInFromWs(TQue<QuePosition::VECIN, BUFFER_NUM>& inQue, const int64_t copyCount,
                                        const int64_t offset);
    __aicore__ inline void Copy2Ws(TQue<QuePosition::VECOUT, BUFFER_NUM>& outQue, const int64_t copyCount,
                                   const int64_t offset);
    __aicore__ inline void CopyWs2Out(TQue<QuePosition::VECOUT, BUFFER_NUM>& outQue, const int64_t copyCount,
                                      const int64_t offset);

   private:
    TPipe* pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQueue1;
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQueue2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yOutQueue1;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yOutQueue2;
    TBuf<TPosition::VECCALC> smallFloatCastResBuf;
    TBuf<TPosition::VECCALC> largeFloatCastResBuf;
    event_t MTE3ToMTE2Event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    LocalTensor<float> smallFloatTenosr;
    LocalTensor<float> largeFloatTenosr;
};

// init used buffer
template <typename T>
__aicore__ inline void PadV4GradPadW<T>::InitBuffer(TPipe* inputPipe) {
    pipe = inputPipe;
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        pipe->InitBuffer(xInQueue1, BUFFER_NUM, CAL_COUNT * sizeof(T) * 2);
        pipe->InitBuffer(yOutQueue1, BUFFER_NUM, CAL_COUNT * sizeof(T) * 2);
        pipe->InitBuffer(xInQueue2, BUFFER_NUM, this->ubFactorElement * sizeof(T));
        pipe->InitBuffer(yOutQueue2, BUFFER_NUM, this->ubFactorElement * sizeof(T));
        pipe->InitBuffer(smallFloatCastResBuf, CAL_COUNT * sizeof(float) * 2);
        pipe->InitBuffer(largeFloatCastResBuf, this->ubFactorElement * sizeof(float));
    } else {
        pipe->InitBuffer(xInQueue1, BUFFER_NUM, CAL_COUNT * sizeof(T) * 2);
        pipe->InitBuffer(yOutQueue1, BUFFER_NUM, CAL_COUNT * sizeof(T) * 2);
        pipe->InitBuffer(xInQueue2, BUFFER_NUM, this->ubFactorElement * sizeof(T));
        pipe->InitBuffer(yOutQueue2, BUFFER_NUM, this->ubFactorElement * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void PadV4GradPadW<T>::CopyGm2UB(TQue<QuePosition::VECIN, BUFFER_NUM>& inQue, const int64_t offset1,
                                                   const int64_t offset2, const int64_t copyCount) {
    int32_t alignCopyCount = this->CeilAlign(copyCount, this->perBlockCount);
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {true, 0, (uint8_t)(alignCopyCount - copyCount), (T)0};
    LocalTensor<T> dataLocal = inQue.AllocTensor<T>();
    DataCopyPad(dataLocal[0], this->mGmX[offset1], copyParams, padParams);
    DataCopyPad(dataLocal[copyCount], this->mGmX[offset2], copyParams, padParams);
    inQue.EnQue(dataLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadW<T>::CopyInFromWs(TQue<QuePosition::VECIN, BUFFER_NUM>& inQue,
                                                      const int64_t copyCount, const int64_t offset) {
    int32_t alignCopyCount = this->CeilAlign(copyCount, this->perBlockCount);
    LocalTensor<T> xLocal = inQue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {true, 0, (uint8_t)(alignCopyCount - copyCount), (T)0};
    DataCopyPad(xLocal, this->mGmWorkspace[offset], copyParams, padParams);
    inQue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadW<T>::CopyIn(const int64_t copyCount, const int64_t offset) {
    int32_t alignCopyCount = this->CeilAlign(copyCount, this->perBlockCount);
    LocalTensor<T> xLocal = xInQueue2.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {true, 0, (uint8_t)(alignCopyCount - copyCount), (T)0};
    DataCopyPad(xLocal, this->mGmX[offset], copyParams, padParams);
    xInQueue2.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadW<T>::Compute(TQue<QuePosition::VECIN, BUFFER_NUM>& inQue,
                                                 TQue<QuePosition::VECOUT, BUFFER_NUM>& outQue,
                                                 const int64_t copyCount) {
    int32_t alignCopyCount = this->CeilAlign(copyCount, this->perBlockCount);
    LocalTensor<T> yLocal = outQue.AllocTensor<T>();
    LocalTensor<T> xLocal = inQue.DeQue<T>();
    DataCopy(yLocal, xLocal, alignCopyCount);
    outQue.EnQue(yLocal);
    inQue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadW<T>::Copy2Ws(TQue<QuePosition::VECOUT, BUFFER_NUM>& outQue,
                                                 const int64_t copyCount, const int64_t offset) {
    int32_t alignCopyCount = this->CeilAlign(copyCount, this->perBlockCount);
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = outQue.DeQue<T>();
    DataCopyPad(this->mGmWorkspace[offset], yLocal, copyParams);
    outQue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadW<T>::CopyWs2Out(TQue<QuePosition::VECOUT, BUFFER_NUM>& outQue,
                                                    const int64_t copyCount, const int64_t offset) {
    int32_t alignCopyCount = this->CeilAlign(copyCount, this->perBlockCount);
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = outQue.DeQue<T>();
    DataCopyPad(this->mGmY[offset], yLocal, copyParams);
    outQue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadW<T>::CopyOut(const int64_t copyCount, const int64_t offset) {
    int32_t alignCopyCount = this->CeilAlign(copyCount, this->perBlockCount);
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = yOutQueue2.DeQue<T>();
    DataCopyPad(this->mGmY[offset], yLocal, copyParams);
    yOutQueue2.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadW<T>::ComputeWGrad(TQue<QuePosition::VECIN, BUFFER_NUM>& inQue,
                                                      TQue<QuePosition::VECOUT, BUFFER_NUM>& outQue,
                                                      const int32_t calCount) {
    LocalTensor<T> xLocal = inQue.DeQue<T>();
    LocalTensor<T> yLocal = outQue.AllocTensor<T>();
    T val1;
    T val2;
    T val3;
    T val4;
    T tmp1;
    T tmp2;
    // compute grad
    for (size_t i = 0; i < this->wPad1; i++) {
        val1 = xLocal.GetValue(i);
        val2 = xLocal.GetValue(2 * this->wPad1 - i);
        if constexpr (AscendC::IsSameType<T, half>::value) {
            tmp1 = (T)((float)val1 + (float)val2);
            xLocal.SetValue(2 * this->wPad1 - i, tmp1);
        } else {
            xLocal.SetValue(2 * this->wPad1 - i, val1 + val2);
        }
    }
    for (size_t i = 0; i < this->wPad2; i++) {
        val3 = xLocal.GetValue(2 * calCount - 1 - i);
        val4 = xLocal.GetValue(2 * calCount - 1 - 2 * this->wPad2 + i);
        if constexpr (AscendC::IsSameType<T, half>::value) {
            tmp2 = (T)((float)val3 + (float)val4);
            xLocal.SetValue(2 * calCount - 1 - 2 * this->wPad2 + i, tmp2);
        } else {
            xLocal.SetValue(2 * calCount - 1 - 2 * this->wPad2 + i, val3 + val4);
        }
    }
    DataCopy(yLocal, xLocal, 2 * calCount);
    outQue.EnQue(yLocal);
    inQue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadW<T>::Bf16ComputeWGrad(TQue<QuePosition::VECIN, BUFFER_NUM>& inQue,
                                                          TQue<QuePosition::VECOUT, BUFFER_NUM>& outQue,
                                                          const int32_t calCount, const int32_t castCount,
                                                          LocalTensor<float> castTensor) {
    LocalTensor<T> xLocal = inQue.DeQue<T>();
    LocalTensor<T> yLocal = outQue.AllocTensor<T>();
    float val1;
    float val2;
    float val3;
    float val4;
    Cast(castTensor, xLocal, RoundMode::CAST_NONE, castCount);
    // compute grad
    for (size_t i = 0; i < this->wPad1; i++) {
        val1 = castTensor.GetValue(i);
        val2 = castTensor.GetValue(2 * this->wPad1 - i);
        castTensor.SetValue(2 * this->wPad1 - i, val1 + val2);
    }
    for (size_t i = 0; i < this->wPad2; i++) {
        val3 = castTensor.GetValue(2 * calCount - 1 - i);
        val4 = castTensor.GetValue(2 * calCount - 1 - 2 * this->wPad2 + i);
        castTensor.SetValue(2 * calCount - 1 - 2 * this->wPad2 + i, val3 + val4);
    }
    Cast(yLocal, castTensor, RoundMode::CAST_ROUND, castCount);
    outQue.EnQue(yLocal);
    inQue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadW<T>::Process() {
    this->batchStride = this->width;
    this->outBatchStride = this->outWidth;
    int64_t gmXOffset = 0;
    int64_t gmXOffset1;
    int64_t gmXOffset2;
    int64_t gmXOffset3;
    int64_t gmYOffset1;
    int64_t gmYOffset2;
    int64_t gmYOffset3;
    int64_t workspaceOffset1;
    int64_t workspaceOffset2;
    int64_t workspaceOffset3;
    int64_t copyCount = 0;
    int64_t calCount = CAL_COUNT;
    int64_t largePadcalCount = this->wPadCopyCount;
    int64_t dataCountOneLine = 0;
    uint32_t copyTimesOneLine = this->CeilDiv(dataCountOneLine, this->ubFactorElement);
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        smallFloatTenosr = smallFloatCastResBuf.Get<float>();
        largeFloatTenosr = largeFloatCastResBuf.Get<float>();
        if (this->wPad1 < W_PAD_LOWER_LIMIT && this->wPad2 < W_PAD_LOWER_LIMIT) {
            dataCountOneLine = this->width - 2 * calCount;
            copyTimesOneLine = this->CeilDiv(dataCountOneLine, this->ubFactorElement);
            for (size_t loop = 0; loop < this->loopNC; loop++) {
                copyCount = this->ubFactorElement;
                gmXOffset1 = loop * this->batchStride + this->ncOffset * this->batchStride;
                gmXOffset2 = (this->width - calCount) + loop * this->batchStride + this->ncOffset * this->batchStride;
                workspaceOffset1 = this->blockIdx * 2 * calCount;
                workspaceOffset2 = this->wPad1 + this->blockIdx * 2 * calCount;
                workspaceOffset3 = calCount + this->blockIdx * 2 * calCount;
                gmYOffset1 = loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
                gmYOffset3 = (this->outWidth - calCount + this->wPad2) + loop * this->outBatchStride +
                             this->ncOffset * this->outBatchStride;
                CopyGm2UB(xInQueue1, gmXOffset1, gmXOffset2, calCount);
                Bf16ComputeWGrad(xInQueue1, yOutQueue1, calCount, 2 * CAL_COUNT, smallFloatTenosr);
                Copy2Ws(yOutQueue1, 2 * calCount, workspaceOffset1);
                set_flag(PIPE_MTE3, PIPE_MTE2, MTE3ToMTE2Event);
                wait_flag(PIPE_MTE3, PIPE_MTE2, MTE3ToMTE2Event);
                CopyInFromWs(xInQueue1, calCount - this->wPad1, workspaceOffset2);
                Compute(xInQueue1, yOutQueue1, calCount - this->wPad1);
                CopyWs2Out(yOutQueue1, calCount - this->wPad1, gmYOffset1);
                CopyInFromWs(xInQueue1, calCount - this->wPad2, workspaceOffset3);
                Compute(xInQueue1, yOutQueue1, calCount - this->wPad2);
                CopyWs2Out(yOutQueue1, calCount - this->wPad2, gmYOffset3);
                for (size_t time = 0; time < copyTimesOneLine; time++) {
                    if (time == copyTimesOneLine - 1) {
                        copyCount = dataCountOneLine - (copyTimesOneLine - 1) * this->ubFactorElement;
                    }
                    gmXOffset3 = calCount + time * this->ubFactorElement + loop * this->batchStride +
                                 this->ncOffset * this->batchStride;
                    gmYOffset2 = (calCount - this->wPad1) + time * this->ubFactorElement +
                                 loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
                    CopyIn(copyCount, gmXOffset3);
                    Compute(xInQueue2, yOutQueue2, copyCount);
                    CopyOut(copyCount, gmYOffset2);
                }
            }
        } else {
            dataCountOneLine = this->width - 2 * largePadcalCount;
            copyTimesOneLine = this->CeilDiv(dataCountOneLine, this->ubFactorElement);
            for (size_t loop = 0; loop < this->loopNC; loop++) {
                copyCount = this->ubFactorElement;
                gmXOffset1 = loop * this->batchStride + this->ncOffset * this->batchStride;
                gmXOffset2 =
                    (this->width - largePadcalCount) + loop * this->batchStride + this->ncOffset * this->batchStride;
                workspaceOffset1 = this->blockIdx * 2 * largePadcalCount;
                workspaceOffset2 = this->wPad1 + this->blockIdx * 2 * largePadcalCount;
                workspaceOffset3 = largePadcalCount + this->blockIdx * 2 * largePadcalCount;
                gmYOffset1 = loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
                gmYOffset3 = (this->outWidth - largePadcalCount + this->wPad2) + loop * this->outBatchStride +
                             this->ncOffset * this->outBatchStride;
                CopyGm2UB(xInQueue2, gmXOffset1, gmXOffset2, largePadcalCount);
                Bf16ComputeWGrad(xInQueue2, yOutQueue2, largePadcalCount, this->ubFactorElement, largeFloatTenosr);
                Copy2Ws(yOutQueue2, 2 * largePadcalCount, workspaceOffset1);
                set_flag(PIPE_MTE3, PIPE_MTE2, MTE3ToMTE2Event);
                wait_flag(PIPE_MTE3, PIPE_MTE2, MTE3ToMTE2Event);
                CopyInFromWs(xInQueue2, largePadcalCount - this->wPad1, workspaceOffset2);
                Compute(xInQueue2, yOutQueue2, largePadcalCount - this->wPad1);
                CopyWs2Out(yOutQueue2, largePadcalCount - this->wPad1, gmYOffset1);
                CopyInFromWs(xInQueue2, largePadcalCount - this->wPad2, workspaceOffset3);
                Compute(xInQueue2, yOutQueue2, largePadcalCount - this->wPad2);
                CopyWs2Out(yOutQueue2, largePadcalCount - this->wPad2, gmYOffset3);
                for (size_t time = 0; time < copyTimesOneLine; time++) {
                    if (time == copyTimesOneLine - 1) {
                        copyCount = dataCountOneLine - (copyTimesOneLine - 1) * this->ubFactorElement;
                    }
                    gmXOffset3 = largePadcalCount + time * this->ubFactorElement + loop * this->batchStride +
                                 this->ncOffset * this->batchStride;
                    gmYOffset2 = (largePadcalCount - this->wPad1) + time * this->ubFactorElement +
                                 loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
                    CopyIn(copyCount, gmXOffset3);
                    Compute(xInQueue2, yOutQueue2, copyCount);
                    CopyOut(copyCount, gmYOffset2);
                }
            }
        }
    } else {
        if (this->wPad1 < W_PAD_LOWER_LIMIT && this->wPad2 < W_PAD_LOWER_LIMIT) {
            dataCountOneLine = this->width - 2 * calCount;
            copyTimesOneLine = this->CeilDiv(dataCountOneLine, this->ubFactorElement);
            for (size_t loop = 0; loop < this->loopNC; loop++) {
                copyCount = this->ubFactorElement;
                gmXOffset1 = loop * this->batchStride + this->ncOffset * this->batchStride;
                gmXOffset2 = (this->width - calCount) + loop * this->batchStride + this->ncOffset * this->batchStride;
                workspaceOffset1 = this->blockIdx * 2 * calCount;
                workspaceOffset2 = this->wPad1 + this->blockIdx * 2 * calCount;
                workspaceOffset3 = calCount + this->blockIdx * 2 * calCount;
                gmYOffset1 = loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
                gmYOffset3 = (this->outWidth - calCount + this->wPad2) + loop * this->outBatchStride +
                             this->ncOffset * this->outBatchStride;
                CopyGm2UB(xInQueue1, gmXOffset1, gmXOffset2, calCount);
                ComputeWGrad(xInQueue1, yOutQueue1, calCount);
                Copy2Ws(yOutQueue1, 2 * calCount, workspaceOffset1);
                set_flag(PIPE_MTE3, PIPE_MTE2, MTE3ToMTE2Event);
                wait_flag(PIPE_MTE3, PIPE_MTE2, MTE3ToMTE2Event);
                CopyInFromWs(xInQueue1, calCount - this->wPad1, workspaceOffset2);
                Compute(xInQueue1, yOutQueue1, calCount - this->wPad1);
                CopyWs2Out(yOutQueue1, calCount - this->wPad1, gmYOffset1);
                CopyInFromWs(xInQueue1, calCount - this->wPad2, workspaceOffset3);
                Compute(xInQueue1, yOutQueue1, calCount - this->wPad2);
                CopyWs2Out(yOutQueue1, calCount - this->wPad2, gmYOffset3);
                for (size_t time = 0; time < copyTimesOneLine; time++) {
                    if (time == copyTimesOneLine - 1) {
                        copyCount = dataCountOneLine - (copyTimesOneLine - 1) * this->ubFactorElement;
                    }
                    gmXOffset3 = calCount + time * this->ubFactorElement + loop * this->batchStride +
                                 this->ncOffset * this->batchStride;
                    gmYOffset2 = (calCount - this->wPad1) + time * this->ubFactorElement +
                                 loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
                    CopyIn(copyCount, gmXOffset3);
                    Compute(xInQueue2, yOutQueue2, copyCount);
                    CopyOut(copyCount, gmYOffset2);
                }
            }
        } else {
            dataCountOneLine = this->width - 2 * largePadcalCount;
            copyTimesOneLine = this->CeilDiv(dataCountOneLine, this->ubFactorElement);
            for (size_t loop = 0; loop < this->loopNC; loop++) {
                copyCount = this->ubFactorElement;
                gmXOffset1 = loop * this->batchStride + this->ncOffset * this->batchStride;
                gmXOffset2 =
                    (this->width - largePadcalCount) + loop * this->batchStride + this->ncOffset * this->batchStride;
                workspaceOffset1 = this->blockIdx * 2 * largePadcalCount;
                workspaceOffset2 = this->wPad1 + this->blockIdx * 2 * largePadcalCount;
                workspaceOffset3 = largePadcalCount + this->blockIdx * 2 * largePadcalCount;
                gmYOffset1 = loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
                gmYOffset3 = (this->outWidth - largePadcalCount + this->wPad2) + loop * this->outBatchStride +
                             this->ncOffset * this->outBatchStride;
                CopyGm2UB(xInQueue2, gmXOffset1, gmXOffset2, largePadcalCount);
                ComputeWGrad(xInQueue2, yOutQueue2, largePadcalCount);
                Copy2Ws(yOutQueue2, 2 * largePadcalCount, workspaceOffset1);
                set_flag(PIPE_MTE3, PIPE_MTE2, MTE3ToMTE2Event);
                wait_flag(PIPE_MTE3, PIPE_MTE2, MTE3ToMTE2Event);
                CopyInFromWs(xInQueue2, largePadcalCount - this->wPad1, workspaceOffset2);
                Compute(xInQueue2, yOutQueue2, largePadcalCount - this->wPad1);
                CopyWs2Out(yOutQueue2, largePadcalCount - this->wPad1, gmYOffset1);
                CopyInFromWs(xInQueue2, largePadcalCount - this->wPad2, workspaceOffset3);
                Compute(xInQueue2, yOutQueue2, largePadcalCount - this->wPad2);
                CopyWs2Out(yOutQueue2, largePadcalCount - this->wPad2, gmYOffset3);
                for (size_t time = 0; time < copyTimesOneLine; time++) {
                    if (time == copyTimesOneLine - 1) {
                        copyCount = dataCountOneLine - (copyTimesOneLine - 1) * this->ubFactorElement;
                    }
                    gmXOffset3 = largePadcalCount + time * this->ubFactorElement + loop * this->batchStride +
                                 this->ncOffset * this->batchStride;
                    gmYOffset2 = (largePadcalCount - this->wPad1) + time * this->ubFactorElement +
                                 loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
                    CopyIn(copyCount, gmXOffset3);
                    Compute(xInQueue2, yOutQueue2, copyCount);
                    CopyOut(copyCount, gmYOffset2);
                }
            }
        }
    }
}
#endif  // _PAD_V4_GRAD_W_PAD_H_