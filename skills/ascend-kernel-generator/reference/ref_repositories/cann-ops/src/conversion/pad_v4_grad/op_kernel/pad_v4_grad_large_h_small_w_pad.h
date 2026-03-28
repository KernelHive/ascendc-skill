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
 * \file pad_v4_grad_large_h_small_w_pad.h
 * \brief
 */
#ifndef _PAD_V4_GRAD_LARGE_H_SMALL_W_PAD_H_
#define _PAD_V4_GRAD_LARGE_H_SMALL_W_PAD_H_

#include "pad_v4_grad_base.h"

template <typename T>
class PadV4GradLargeHSmallW : public PadV4GradBase<T> {
   public:
    __aicore__ inline PadV4GradLargeHSmallW(){};
    __aicore__ inline void InitBuffer(TPipe *inputPipe);
    __aicore__ inline void CopyGm2UB(const int32_t batchIdx, const int32_t flag);
    __aicore__ inline void CopyOut2Gm(const int32_t batchIdx, const int32_t cycles, const int32_t transBlkIdx);
    __aicore__ inline void CopyGmAndWs2UB1(const int32_t batchIdx);
    __aicore__ inline void CopyGmAndWorkspace2UB2(const int32_t transBlkIdx, const int32_t transTimes,
                                                  const int32_t cycles, const int32_t batchIdx);
    __aicore__ inline void CopyOut2Ws(const int64_t calCount, const int32_t flag);
    __aicore__ inline void ComputeHGrad(const int32_t calCount, const int32_t flag);
    __aicore__ inline void implTransposeAndCompute(const int64_t transCount);
    __aicore__ inline void Process();

   private:
    TPipe *pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, 1> xInQueue;
    TQue<QuePosition::VECOUT, 1> yOutQueue;
    TQue<QuePosition::VECOUT, 1> transposeQue;

    event_t MTE3ToMTE2Event;
};

// init used buffer
template <typename T>
__aicore__ inline void PadV4GradLargeHSmallW<T>::InitBuffer(TPipe *inputPipe) {
    pipe = inputPipe;
    pipe->InitBuffer(xInQueue, 1, this->ubFactorElement * SMALL_WIDTH_LIMIT * sizeof(T));
    pipe->InitBuffer(yOutQueue, 1, this->ubFactorElement * SMALL_WIDTH_LIMIT * sizeof(T));
    pipe->InitBuffer(transposeQue, 1, this->ubFactorElement * SMALL_WIDTH_LIMIT * sizeof(T));
}

template <typename T>
__aicore__ inline void PadV4GradLargeHSmallW<T>::CopyGm2UB(const int32_t batchIdx, const int32_t flag) {
    int64_t gmXOffset;
    int32_t alignCopyCount = this->CeilAlign(this->width, this->perBlockCount);
    DataCopyExtParams copyParams{1, (uint32_t)(this->width * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {true, 0, (uint8_t)(alignCopyCount - this->width), (T)0};
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    if (flag == 0) {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS; i++) {
            gmXOffset = i * this->width + batchIdx * this->batchStride + this->ncOffset * this->batchStride;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmX[gmXOffset], copyParams, padParams);
        }
    } else {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS; i++) {
            gmXOffset = (this->height - COPY_ROWS_AND_COLS + i) * this->width + batchIdx * this->batchStride +
                        this->ncOffset * this->batchStride;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmX[gmXOffset], copyParams, padParams);
        }
    }
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV4GradLargeHSmallW<T>::CopyOut2Gm(const int32_t batchIdx, const int32_t cycles,
                                                            const int32_t transBlkIdx) {
    int64_t gmYOffset1 = 0;
    int64_t gmYOffset2 = 0;
    DataCopyExtParams copyParams{1, (uint32_t)(this->outWidth * sizeof(T)), 0, 0, 0};
    LocalTensor<T> transposeData = transposeQue.DeQue<T>();
    if (cycles <= COPY_ROWS_AND_COLS - this->hPad2) {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
            gmYOffset1 = this->outWidth * (this->outHeight - (COPY_ROWS_AND_COLS - this->hPad2) + i) +
                         batchIdx * this->outBatchStride + this->ncOffset * this->outBatchStride;
            DataCopyPad(this->mGmY[gmYOffset1], transposeData[i * SMALL_WIDTH_LIMIT], copyParams);
        }
    } else {
        for (size_t i = 0; i < cycles; i++) {
            gmYOffset2 = this->outWidth * (i + transBlkIdx * this->ubFactorElement) + batchIdx * this->outBatchStride +
                         this->ncOffset * this->outBatchStride;
            DataCopyPad(this->mGmY[gmYOffset2], transposeData[i * SMALL_WIDTH_LIMIT], copyParams);
        }
    }

    transposeQue.FreeTensor(transposeData);
}

template <typename T>
__aicore__ inline void PadV4GradLargeHSmallW<T>::CopyGmAndWs2UB1(const int32_t batchIdx) {
    DataCopyExtParams copyParams{1, (uint32_t)(this->width * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    int64_t workspaceOffset1;
    int64_t workspaceOffset2;
    int64_t xGmOffset;
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad1; i++) {
        workspaceOffset1 = i * this->width + this->blockIdx * this->workspacePerCore;
        DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmWorkspace[workspaceOffset1], copyParams, padParams);
    }
    for (size_t i = COPY_ROWS_AND_COLS; i < this->height - COPY_ROWS_AND_COLS; i++) {
        xGmOffset = i * this->width + batchIdx * this->batchStride + this->ncOffset * this->batchStride;
        DataCopyPad(xLocal[(i - this->hPad1) * SMALL_WIDTH_LIMIT], this->mGmX[xGmOffset], copyParams, padParams);
    }
    for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
        workspaceOffset2 =
            (i + COPY_ROWS_AND_COLS - this->hPad1) * this->width + this->blockIdx * this->workspacePerCore;
        DataCopyPad(xLocal[(this->outHeight - (COPY_ROWS_AND_COLS - this->hPad2) + i) * SMALL_WIDTH_LIMIT],
                    this->mGmWorkspace[workspaceOffset2], copyParams, padParams);
    }
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV4GradLargeHSmallW<T>::CopyGmAndWorkspace2UB2(const int32_t transBlkIdx,
                                                                        const int32_t transTimes, const int32_t cycles,
                                                                        const int32_t batchIdx) {
    DataCopyExtParams copyParams{1, (uint32_t)(this->width * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    int64_t workspaceOffset1;
    int64_t workspaceOffset2;
    int64_t workspaceOffset3;
    int64_t xGmOffset1;
    int64_t xGmOffset2;
    int64_t xGmOffset3;
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();

    if (transBlkIdx == 0) {
        for (size_t i = 0; i < this->ubFactorElement; i++) {
            xGmOffset1 =
                (i + this->hPad1) * this->width + batchIdx * this->batchStride + this->ncOffset * this->batchStride;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmX[xGmOffset1], copyParams, padParams);
        }
        pipe_barrier(PIPE_MTE2);
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad1; i++) {
            workspaceOffset1 = i * this->width + this->blockIdx * this->workspacePerCore;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmWorkspace[workspaceOffset1], copyParams, padParams);
        }

    } else if (transBlkIdx > 0 && transBlkIdx < transTimes - 1) {
        for (size_t i = 0; i < this->ubFactorElement; i++) {
            xGmOffset2 = (this->ubFactorElement * transBlkIdx + this->hPad1 + i) * this->width +
                         batchIdx * this->batchStride + this->ncOffset * this->batchStride;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmX[xGmOffset2], copyParams, padParams);
        }
    } else if (transBlkIdx == transTimes - 1) {
        if (cycles <= COPY_ROWS_AND_COLS - this->hPad2) {
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
                workspaceOffset2 =
                    (i + COPY_ROWS_AND_COLS - this->hPad1) * this->width + this->blockIdx * this->workspacePerCore;
                DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmWorkspace[workspaceOffset2], copyParams,
                            padParams);
            }
        } else {
            for (size_t i = 0; i < cycles; i++) {
                xGmOffset3 = (i + (transTimes - 1) * this->ubFactorElement + this->hPad1) * this->width +
                             batchIdx * this->batchStride + this->ncOffset * this->batchStride;
                DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmX[xGmOffset3], copyParams, padParams);
            }
            pipe_barrier(PIPE_MTE2);
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
                workspaceOffset3 =
                    (i + COPY_ROWS_AND_COLS - this->hPad1) * this->width + this->blockIdx * this->workspacePerCore;
                DataCopyPad(xLocal[(cycles - (COPY_ROWS_AND_COLS - this->hPad2) + i) * SMALL_WIDTH_LIMIT],
                            this->mGmWorkspace[workspaceOffset3], copyParams, padParams);
            }
        }
    }

    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV4GradLargeHSmallW<T>::CopyOut2Ws(const int64_t calCount, const int32_t flag) {
    int64_t workspaceOffset;
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    if (flag == 0) {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad1; i++) {
            workspaceOffset = i * this->width + this->blockIdx * this->workspacePerCore;
            DataCopyPad(this->mGmWorkspace[workspaceOffset], yLocal[i * SMALL_WIDTH_LIMIT], copyParams);
        }
    } else {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
            workspaceOffset =
                (COPY_ROWS_AND_COLS - this->hPad1 + i) * this->width + this->blockIdx * this->workspacePerCore;
            DataCopyPad(this->mGmWorkspace[workspaceOffset], yLocal[i * SMALL_WIDTH_LIMIT], copyParams);
        }
    }
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradLargeHSmallW<T>::implTransposeAndCompute(const int64_t transCount) {
    uint32_t loopTimes = this->CeilDiv(transCount, TRANSDATA_BASE_H);
    uint64_t xSrcLocalList0[16];
    uint64_t xDstLocalList0[16];
    uint64_t xSrcLocalList1[16];
    uint64_t xDstLocalList1[16];
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> transposeData = transposeQue.AllocTensor<T>();
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = 1;
    transDataParams.dstRepStride = 0;
    transDataParams.srcRepStride = 0;
    if constexpr (AscendC::IsSameType<T, half>::value) {
        for (size_t time = 0; time < SMALL_WIDTH_LIMIT / HALF_BLOCK_NUM; time++) {
            for (int i = 0; i < HALF_BLOCK_NUM; i++) {
                xSrcLocalList0[i] = (uint64_t)(xLocal[SMALL_WIDTH_LIMIT * i + HALF_BLOCK_NUM * time].GetPhyAddr());
                xDstLocalList0[i] =
                    (uint64_t)(transposeData[this->ubFactorElement * i + time * this->ubFactorElement * 16]
                                   .GetPhyAddr());
            }
            transDataParams.repeatTimes = loopTimes;
            transDataParams.srcRepStride = TRANSDATA_BASE_H * SMALL_WIDTH_LIMIT * sizeof(T) / DATA_BLOCK_BYTES;
            transDataParams.dstRepStride = 1;
            TransDataTo5HD<T>(xDstLocalList0, xSrcLocalList0, transDataParams);
        }

        for (size_t i = 0; i < this->wPad1; i++) {
            Add(transposeData[(2 * this->wPad1 - i) * this->ubFactorElement], transposeData[i * this->ubFactorElement],
                transposeData[(2 * this->wPad1 - i) * this->ubFactorElement], this->ubFactorElement);
        }
        for (size_t i = 0; i < this->wPad2; i++) {
            Add(transposeData[(this->width - 2 * this->wPad2 - 1 + i) * this->ubFactorElement],
                transposeData[(this->width - 1 - i) * this->ubFactorElement],
                transposeData[(this->width - 2 * this->wPad2 - 1 + i) * this->ubFactorElement], this->ubFactorElement);
        }
        DataCopy(transposeData, transposeData[this->wPad1 * this->ubFactorElement],
                 this->outWidth * this->ubFactorElement);
        for (size_t time = 0; time < loopTimes; time++) {
            for (int i = 0; i < HALF_BLOCK_NUM; i++) {
                xSrcLocalList1[i] =
                    (uint64_t)(transposeData[this->ubFactorElement * i + time * HALF_BLOCK_NUM].GetPhyAddr());
                xDstLocalList1[i] =
                    (uint64_t)(xLocal[SMALL_WIDTH_LIMIT * i + time * HALF_BLOCK_NUM * SMALL_WIDTH_LIMIT].GetPhyAddr());
            }
            transDataParams.repeatTimes = SMALL_WIDTH_LIMIT / TRANSDATA_BASE_H;
            transDataParams.srcRepStride = this->ubFactorElement;
            transDataParams.dstRepStride = 1;
            TransDataTo5HD<T>(xDstLocalList1, xSrcLocalList1, transDataParams);
        }
        DataCopy(transposeData, xLocal, SMALL_WIDTH_LIMIT * this->ubFactorElement);
        xInQueue.FreeTensor(xLocal);
        transposeQue.EnQue(transposeData);
    } else {
        for (size_t time = 0; time < SMALL_WIDTH_LIMIT / FLOAT_BLOCK_NUM; time++) {
            for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
                xSrcLocalList0[i] = (uint64_t)(xLocal[SMALL_WIDTH_LIMIT * i + FLOAT_BLOCK_NUM * time].GetPhyAddr());
            }
            for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
                xDstLocalList0[2 * i] = (uint64_t)(transposeData[i * this->ubFactorElement +
                                                                 FLOAT_BLOCK_NUM * this->ubFactorElement * time]
                                                       .GetPhyAddr());
                xDstLocalList0[2 * i + 1] =
                    (uint64_t)(transposeData[i * this->ubFactorElement +
                                             FLOAT_BLOCK_NUM * this->ubFactorElement * time + FLOAT_BLOCK_NUM]
                                   .GetPhyAddr());
            }
            transDataParams.repeatTimes = loopTimes;
            transDataParams.srcRepStride = TRANSDATA_BASE_H * SMALL_WIDTH_LIMIT * sizeof(T) / DATA_BLOCK_BYTES;
            transDataParams.dstRepStride = TRANSDATA_BASE_H / FLOAT_BLOCK_NUM;
            TransDataTo5HD<T>(xDstLocalList0, xSrcLocalList0, transDataParams);
        }
        for (size_t i = 0; i < this->wPad1; i++) {
            Add(transposeData[(2 * this->wPad1 - i) * this->ubFactorElement], transposeData[i * this->ubFactorElement],
                transposeData[(2 * this->wPad1 - i) * this->ubFactorElement], this->ubFactorElement);
        }
        for (size_t i = 0; i < this->wPad2; i++) {
            Add(transposeData[(this->width - 2 * this->wPad2 - 1 + i) * this->ubFactorElement],
                transposeData[(this->width - 1 - i) * this->ubFactorElement],
                transposeData[(this->width - 2 * this->wPad2 - 1 + i) * this->ubFactorElement], this->ubFactorElement);
        }
        DataCopy(transposeData, transposeData[this->wPad1 * this->ubFactorElement],
                 this->outWidth * this->ubFactorElement);

        for (size_t time = 0; time < this->ubFactorElement / FLOAT_BLOCK_NUM; time++) {
            for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
                xSrcLocalList1[i] =
                    (uint64_t)(transposeData[this->ubFactorElement * i + time * FLOAT_BLOCK_NUM].GetPhyAddr());
            }
            for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
                xDstLocalList1[2 * i] =
                    (uint64_t)(xLocal[SMALL_WIDTH_LIMIT * i + time * SMALL_WIDTH_LIMIT * FLOAT_BLOCK_NUM]
                                   .GetPhyAddr());
                xDstLocalList1[2 * i + 1] =
                    (uint64_t)(xLocal[SMALL_WIDTH_LIMIT * i + time * SMALL_WIDTH_LIMIT * FLOAT_BLOCK_NUM +
                                      FLOAT_BLOCK_NUM]
                                   .GetPhyAddr());
            }
            transDataParams.repeatTimes = SMALL_WIDTH_LIMIT / TRANSDATA_BASE_H;
            transDataParams.srcRepStride = 2 * this->ubFactorElement;
            transDataParams.dstRepStride = TRANSDATA_BASE_H / FLOAT_BLOCK_NUM;
            TransDataTo5HD<T>(xDstLocalList1, xSrcLocalList1, transDataParams);
        }
        DataCopy(transposeData, xLocal, SMALL_WIDTH_LIMIT * this->ubFactorElement);
        xInQueue.FreeTensor(xLocal);
        transposeQue.EnQue(transposeData);
    }
}

template <typename T>
__aicore__ inline void PadV4GradLargeHSmallW<T>::ComputeHGrad(const int32_t calCount, const int32_t flag) {
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    // compute grad
    if (flag == 0) {
        for (size_t i = 0; i < this->hPad1; i++) {
            Add(xLocal[(2 * this->hPad1 - i) * SMALL_WIDTH_LIMIT], xLocal[i * SMALL_WIDTH_LIMIT],
                xLocal[(2 * this->hPad1 - i) * SMALL_WIDTH_LIMIT], calCount);
        }
        DataCopy(yLocal, xLocal[this->hPad1 * SMALL_WIDTH_LIMIT],
                 (COPY_ROWS_AND_COLS - this->hPad1) * SMALL_WIDTH_LIMIT);
    } else {
        for (size_t i = 0; i < this->hPad2; i++) {
            Add(xLocal[(COPY_ROWS_AND_COLS - 2 * this->hPad2 - 1 + i) * SMALL_WIDTH_LIMIT],
                xLocal[(COPY_ROWS_AND_COLS - 1 - i) * SMALL_WIDTH_LIMIT],
                xLocal[(COPY_ROWS_AND_COLS - 2 * this->hPad2 - 1 + i) * SMALL_WIDTH_LIMIT], calCount);
        }
        DataCopy(yLocal, xLocal, (COPY_ROWS_AND_COLS - this->hPad2) * SMALL_WIDTH_LIMIT);
    }
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradLargeHSmallW<T>::Process() {
    int64_t calCount = this->width;
    int64_t cycleTimes = this->ubFactorElement;

    uint32_t transTimesOneCol = this->CeilDiv(this->outHeight, this->ubFactorElement);

    for (size_t loop = 0; loop < this->loopNC; loop++) {
        CopyGm2UB(loop, 0);
        ComputeHGrad(calCount, 0);
        CopyOut2Ws(calCount, 0);
        CopyGm2UB(loop, 1);
        ComputeHGrad(calCount, 1);
        CopyOut2Ws(calCount, 1);

        set_flag(PIPE_MTE3, PIPE_MTE2, MTE3ToMTE2Event);
        wait_flag(PIPE_MTE3, PIPE_MTE2, MTE3ToMTE2Event);
        if (transTimesOneCol == 1) {
            CopyGmAndWs2UB1(loop);
            implTransposeAndCompute(this->ubFactorElement);
            CopyOut2Gm(loop, this->outHeight, 0);
        } else if (transTimesOneCol > 1) {
            for (size_t transBlk = 0; transBlk < transTimesOneCol; transBlk++) {
                cycleTimes = this->ubFactorElement;
                if (transBlk == transTimesOneCol - 1) {
                    cycleTimes = this->outHeight - (transTimesOneCol - 1) * this->ubFactorElement;
                }
                CopyGmAndWorkspace2UB2(transBlk, transTimesOneCol, cycleTimes, loop);
                implTransposeAndCompute(this->ubFactorElement);
                CopyOut2Gm(loop, cycleTimes, transBlk);
            }
        }
    }
}
#endif  // _PAD_V4_GRAD_LARGE_H_SMALL_W_PAD_H_