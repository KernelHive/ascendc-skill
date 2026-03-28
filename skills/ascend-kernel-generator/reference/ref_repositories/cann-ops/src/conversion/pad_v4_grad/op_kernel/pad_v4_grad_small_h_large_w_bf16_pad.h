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
 * \file pad_v4_grad_small_h_large_w_bf16_pad.h
 * \brief
 */
#ifndef _PAD_V4_GRAD_SMALL_H_LARGE_W_BF16_PAD_H_
#define _PAD_V4_GRAD_SMALL_H_LARGE_W_BF16_PAD_H_

#include "pad_v4_grad_base.h"

template <typename T>
class PadV4GradPadSamllHLargeWBf16: public PadV4GradBase<T> {
   public:
    __aicore__ inline PadV4GradPadSamllHLargeWBf16(){};
    __aicore__ inline void InitBuffer(TPipe *inputPipe);
    __aicore__ inline void CopyGm2UB(const int32_t cycleIdx, const int64_t copyCount, const int32_t batchIdx);
    __aicore__ inline void CopyOut2Gm(const int32_t batchIdx, const int32_t cycles, const int32_t flag);
    __aicore__ inline void CopyWs2UB(const int32_t batchIdx, const int64_t copyCount, const int32_t flag);
    __aicore__ inline void CopyOut2Workspace(const int32_t tIdx, const int64_t calCount);
    __aicore__ inline void ComputeHGrad(const int32_t calCount);
    __aicore__ inline void implTransposeAndCompute(const int64_t transCount, const int32_t flag);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(const int32_t copyCount, const int64_t workspaceOffset);
    __aicore__ inline void compute(const int32_t copyCount);
    __aicore__ inline void CopyOut(const int32_t copyCount, const int64_t offset);

   private:
    TPipe *pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, 1> xInQueue;
    TQue<QuePosition::VECOUT, 1> yOutQueue;
    TBuf<TPosition::VECCALC> transposeBuf;
    TBuf<TPosition::VECCALC> floatCastResBuf;
    LocalTensor<float> floatTenosr;
    LocalTensor<float> transposeData;
    event_t MTE3ToMTE2Event;
};

// init used buffer
template <typename T>
__aicore__ inline void PadV4GradPadSamllHLargeWBf16<T>::InitBuffer(TPipe *inputPipe) {
    pipe = inputPipe;
    pipe->InitBuffer(xInQueue, 1, SMALL_HEIGHT_LIMIT * this->ubFactorElement * sizeof(T));
    pipe->InitBuffer(yOutQueue, 1, SMALL_HEIGHT_LIMIT * this->ubFactorElement * sizeof(T));
    pipe->InitBuffer(transposeBuf, SMALL_HEIGHT_LIMIT * this->ubFactorElement * sizeof(float));
    pipe->InitBuffer(floatCastResBuf, SMALL_HEIGHT_LIMIT * this->ubFactorElement * sizeof(float));
}

template <typename T>
__aicore__ inline void PadV4GradPadSamllHLargeWBf16<T>::CopyGm2UB(const int32_t cycleIdx, const int64_t copyCount,
                                                              const int32_t batchIdx) {
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    int32_t alignCopyCount = this->CeilAlign(copyCount, this->perBlockCount);
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(alignCopyCount - copyCount), (T)0};
    int64_t offset = 0;
    for (size_t i = 0; i < this->height; i++) {
        offset = i * this->width + cycleIdx * this->ubFactorElement + batchIdx * this->batchStride + this->ncOffset * this->batchStride;
        DataCopyPad(xLocal[i * this->ubFactorElement], this->mGmX[offset], copyParams, padParams);
    }
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadSamllHLargeWBf16<T>::CopyOut2Gm(const int32_t batchIdx, const int32_t cycles,
                                                               const int32_t flag) {
    int64_t gmYOffset = 0;
    DataCopyExtParams leftCopyParams{1, (uint32_t)((COPY_ROWS_AND_COLS - this->wPad1) * sizeof(T)), 0, 0, 0};
    DataCopyExtParams rightCopyParams{1, (uint32_t)((COPY_ROWS_AND_COLS - this->wPad2) * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    if (flag == 0) {
        for (size_t i = 0; i < cycles; i++) {
            gmYOffset = this->outWidth * i + batchIdx * this->outBatchStride + this->ncOffset * this->outBatchStride;
            DataCopyPad(this->mGmY[gmYOffset], yLocal[i * COPY_ROWS_AND_COLS], leftCopyParams);
        }
    } else {
        for (size_t i = 0; i < cycles; i++) {
            gmYOffset = this->outWidth * (i + 1) - (COPY_ROWS_AND_COLS - this->wPad2) + batchIdx * this->outBatchStride +
                        this->ncOffset * this->outBatchStride;
            DataCopyPad(this->mGmY[gmYOffset], yLocal[i * COPY_ROWS_AND_COLS], rightCopyParams);
        }
    }
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadSamllHLargeWBf16<T>::CopyWs2UB(const int32_t batchIdx, const int64_t copyCount,
                                                              const int32_t flag) {
    DataCopyExtParams copyParams{1, (uint32_t)(COPY_ROWS_AND_COLS * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    int64_t workspaceOffset;
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    if (flag == 0) {
        for (size_t i = 0; i < this->outHeight; i++) {
            workspaceOffset = i * this->width + this->blockIdx * this->workspacePerCore;
            DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmWorkspace[workspaceOffset], copyParams, padParams);
        }
    } else {
        for (size_t i = 0; i < this->outHeight; i++) {
            workspaceOffset = (i + 1) * this->width - COPY_ROWS_AND_COLS + this->blockIdx * this->workspacePerCore;
            DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmWorkspace[workspaceOffset], copyParams, padParams);
        }
    }
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadSamllHLargeWBf16<T>::CopyOut2Workspace(const int32_t tIdx, const int64_t calCount) {
    int64_t workspaceOffset;
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    for (size_t i = 0; i < this->outHeight; i++) {
        workspaceOffset = i * this->width + tIdx * this->ubFactorElement + this->blockIdx * this->workspacePerCore;
        DataCopyPad(this->mGmWorkspace[workspaceOffset], yLocal[i * this->ubFactorElement], copyParams);
    }
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadSamllHLargeWBf16<T>::implTransposeAndCompute(const int64_t transCount,
                                                                            const int32_t flag) {
    uint32_t loopTimes = this->CeilDiv(transCount, TRANSDATA_BASE_H);
    uint64_t xSrcLocalList0[16];
    uint64_t xDstLocalList0[16];
    uint64_t xSrcLocalList1[16];
    uint64_t xDstLocalList1[16];
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = 1;
    transDataParams.dstRepStride = 0;
    transDataParams.srcRepStride = 0;
    Cast(floatTenosr, xLocal, RoundMode::CAST_NONE, SMALL_HEIGHT_LIMIT * this->ubFactorElement);
    for (size_t time = 0; time < COPY_ROWS_AND_COLS / FLOAT_BLOCK_NUM; time++) {
        for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
            xSrcLocalList0[i] = (uint64_t)(floatTenosr[COPY_ROWS_AND_COLS * i + FLOAT_BLOCK_NUM * time].GetPhyAddr());
        }
        for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
            xDstLocalList0[2 * i] =
                (uint64_t)(transposeData[i * this->ubFactorElement + FLOAT_BLOCK_NUM * this->ubFactorElement * time]
                                .GetPhyAddr());
            xDstLocalList0[2 * i + 1] =
                (uint64_t)(transposeData[i * this->ubFactorElement + FLOAT_BLOCK_NUM * this->ubFactorElement * time +
                                            FLOAT_BLOCK_NUM]
                                .GetPhyAddr());
        }
        transDataParams.repeatTimes = loopTimes;
        transDataParams.srcRepStride = TRANSDATA_BASE_H * COPY_ROWS_AND_COLS * sizeof(float) / DATA_BLOCK_BYTES;
        transDataParams.dstRepStride = COPY_ROWS_AND_COLS / FLOAT_BLOCK_NUM;
        TransDataTo5HD<float>(xDstLocalList0, xSrcLocalList0, transDataParams);
    }
    if (flag == 0) {
        for (size_t i = 0; i < this->wPad1; i++) {
            Add(transposeData[(2 * this->wPad1 - i) * this->ubFactorElement], transposeData[i * this->ubFactorElement],
                transposeData[(2 * this->wPad1 - i) * this->ubFactorElement], this->ubFactorElement);
        }
        DataCopy(transposeData, transposeData[this->wPad1 * this->ubFactorElement],
                    (COPY_ROWS_AND_COLS - this->wPad1) * this->ubFactorElement);

    } else {
        for (size_t i = 0; i < this->wPad2; i++) {
            Add(transposeData[(COPY_ROWS_AND_COLS - 2 * this->wPad2 - 1 + i) * this->ubFactorElement],
                transposeData[(COPY_ROWS_AND_COLS - 1 - i) * this->ubFactorElement],
                transposeData[(COPY_ROWS_AND_COLS - 2 * this->wPad2 - 1 + i) * this->ubFactorElement], this->ubFactorElement);
        }
    }
    for (size_t time = 0; time < this->ubFactorElement / FLOAT_BLOCK_NUM; time++) {
        for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
            xSrcLocalList1[i] =
                (uint64_t)(transposeData[this->ubFactorElement * i + time * FLOAT_BLOCK_NUM].GetPhyAddr());
        }
        for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
            xDstLocalList1[2 * i] =
                (uint64_t)(floatTenosr[COPY_ROWS_AND_COLS * i + time * COPY_ROWS_AND_COLS * FLOAT_BLOCK_NUM]
                                .GetPhyAddr());
            xDstLocalList1[2 * i + 1] =
                (uint64_t)(floatTenosr[COPY_ROWS_AND_COLS * i + time * COPY_ROWS_AND_COLS * FLOAT_BLOCK_NUM +
                                    FLOAT_BLOCK_NUM]
                                .GetPhyAddr());
        }
        transDataParams.repeatTimes = 1;
        transDataParams.srcRepStride = 0;
        transDataParams.dstRepStride = 0;
        TransDataTo5HD<float>(xDstLocalList1, xSrcLocalList1, transDataParams);
    }
    Cast(yLocal, floatTenosr, RoundMode::CAST_RINT, SMALL_HEIGHT_LIMIT * this->ubFactorElement);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadSamllHLargeWBf16<T>::ComputeHGrad(const int32_t calCount) {
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    Cast(floatTenosr, xLocal, RoundMode::CAST_NONE, SMALL_HEIGHT_LIMIT * this->ubFactorElement);
    // compute grad
    for (size_t i = 0; i < this->hPad1; i++) {
        Add(floatTenosr[(2 * this->hPad1 - i) * this->ubFactorElement], floatTenosr[i * this->ubFactorElement],
            floatTenosr[(2 * this->hPad1 - i) * this->ubFactorElement], calCount);
    }
    for (size_t i = 0; i < this->hPad2; i++) {
        Add(floatTenosr[(this->height - 2 * this->hPad2 - 1 + i) * this->ubFactorElement], floatTenosr[(this->height - 1 - i) * this->ubFactorElement],
            floatTenosr[(this->height - 2 * this->hPad2 - 1 + i) * this->ubFactorElement], calCount);
    }
    Cast(yLocal, floatTenosr[this->hPad1 * this->ubFactorElement], RoundMode::CAST_RINT, this->outHeight * this->ubFactorElement);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadSamllHLargeWBf16<T>::CopyIn(const int32_t copyCount, const int64_t workspaceOffset) {
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    DataCopyPad(xLocal, this->mGmWorkspace[workspaceOffset], copyParams, padParams);
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadSamllHLargeWBf16<T>::compute(const int32_t copyCount) {
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    uint32_t alignCopyCount = this->CeilAlign(copyCount, this->perBlockCount);
    DataCopy(yLocal, xLocal, alignCopyCount);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadSamllHLargeWBf16<T>::CopyOut(const int32_t copyCount, const int64_t offset) {
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(this->mGmY[offset], yLocal, copyParams);
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadSamllHLargeWBf16<T>::Process() {
    int64_t gmYOffset;
    int64_t workspaceOffset;
    int64_t calCount = this->ubFactorElement;
    uint32_t copyCount = SMALL_HEIGHT_LIMIT * this->ubFactorElement;
    uint32_t copyTimesOneRow = this->CeilDiv(this->width, this->ubFactorElement);
    uint32_t copyMidDataTimes = this->CeilDiv(this->width - 2 * COPY_ROWS_AND_COLS, SMALL_HEIGHT_LIMIT * this->ubFactorElement);
    floatTenosr = floatCastResBuf.Get<float>();
    transposeData = transposeBuf.Get<float>();
    MTE3ToMTE2Event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    for (size_t loop = 0; loop < this->loopNC; loop++) {
        calCount = this->ubFactorElement;
        for (size_t time = 0; time < copyTimesOneRow; time++) {
            if (time == copyTimesOneRow - 1) {
                calCount = this->width - (copyTimesOneRow - 1) * this->ubFactorElement;
            }
            CopyGm2UB(time, calCount, loop);
            ComputeHGrad(calCount);
            CopyOut2Workspace(time, calCount);
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, MTE3ToMTE2Event);
        wait_flag(PIPE_MTE3, PIPE_MTE2, MTE3ToMTE2Event);
        for (size_t i = 0; i < this->outHeight; i++) {
            copyCount = SMALL_HEIGHT_LIMIT * this->ubFactorElement;
            for (size_t j = 0; j < copyMidDataTimes; j++) {
                if (j == copyMidDataTimes - 1) {
                    copyCount =
                        this->width - 2 * COPY_ROWS_AND_COLS - (copyMidDataTimes - 1) * this->ubFactorElement * SMALL_HEIGHT_LIMIT;
                }
                workspaceOffset = COPY_ROWS_AND_COLS + j * this->ubFactorElement * SMALL_HEIGHT_LIMIT + i * this->width +
                                  this->blockIdx * this->workspacePerCore;
                gmYOffset = COPY_ROWS_AND_COLS - this->wPad1 + j * this->ubFactorElement * SMALL_HEIGHT_LIMIT + i * this->outWidth +
                            loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
                CopyIn(copyCount, workspaceOffset);
                compute(this->ubFactorElement * SMALL_HEIGHT_LIMIT);
                CopyOut(copyCount, gmYOffset);
            }
        }
        CopyWs2UB(loop, COPY_ROWS_AND_COLS, 0);
        implTransposeAndCompute(this->ubFactorElement, 0);
        CopyOut2Gm(loop, this->outHeight, 0);
        CopyWs2UB(loop, COPY_ROWS_AND_COLS, 1);
        implTransposeAndCompute(this->ubFactorElement, 1);
        CopyOut2Gm(loop, this->outHeight, 1);
    }
}
#endif  // _PAD_V4_GRAD_SMALL_H_LARGE_W_BF16_PAD_H_