/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file pad_v3_grad_replicate_h_w_large.h
 * \brief
 */
#ifndef _PAD_V3_GRAD_REPLICATE_H_W_LARGE_
#define _PAD_V3_GRAD_REPLICATE_H_W_LARGE_

#include "kernel_operator.h"
#include "pad_v3_grad_replicate_base.h"

using namespace AscendC;

template <typename T>
class PadV3GradReplicateHWLarge {
public:
    __aicore__ inline PadV3GradReplicateHWLarge() {};
    __aicore__ inline void Init(const PadV3GradReplicateTilingData &__restrict tilingData, 
                                GM_ADDR x, GM_ADDR padding, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void InitBuffer(TPipe *inputPipe);
    __aicore__ inline void CopyGm2UB(const int32_t cycleIdx, const int64_t copyCount, const int32_t batchIdx,
                                     const int64_t ncOffset, const int32_t flag);
    __aicore__ inline void CopyOut2Gm(const int32_t batchIdx, const int64_t ncOffset, const int32_t cycles,
                                      const int32_t transBlkIdx, const int32_t flag);
    __aicore__ inline void CopyGmAndWorkspace2UB1(const int32_t batchIdx, const int64_t copyCount,
                                                  const int64_t ncOffset, const int32_t flag);
    __aicore__ inline void CopyGmAndWorkspace2UB2(const int32_t transBlkIdx, const int32_t transTimes,
                                                  const int32_t cycles, const int32_t batchIdx, const int64_t ncOffset,
                                                  const int32_t flag);
    __aicore__ inline void CopyOut2Workspace(const int32_t tIdx, const int64_t calCount, const int32_t flag);
    __aicore__ inline void ComputeHGrad(const int32_t calCount, const int32_t flag);
    __aicore__ inline void ComputeHGradBF16(const int32_t calCount, const int32_t flag);
    __aicore__ inline void ImplTransposeAndCompute(const int64_t transCount, const int32_t flag);
    __aicore__ inline void ImplTransposeAndComputeBF16(const int64_t transCount, const int32_t flag);
    __aicore__ inline void CopyIn(const int32_t copyCount, const int64_t workspaceOffset);
    __aicore__ inline void Compute(const int32_t copyCount);
    __aicore__ inline void CopyOut(const int32_t copyCount, const int64_t offset);
    __aicore__ inline void CopyInFromGm(const int32_t copyCount, const int64_t offset);
    __aicore__ inline void CopyInput2OutGm(const int32_t copyCount, const int64_t offset);
    __aicore__ inline void Process();

private:
    TPipe *pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yOutQueue;
    TBuf<TPosition::VECCALC> transposeBuf;
    TBuf<TPosition::VECCALC> floatCastResBuf;
    LocalTensor<float> floatTensor;
    LocalTensor<float> transposeData;

    uint32_t batch = 0;
    uint32_t ncPerCore = 0;
    uint32_t tailNC = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t alignHeight = 0;
    uint32_t alignWidth = 0;
    uint32_t outHeight = 0;
    uint32_t outWidth = 0;
    uint32_t alignOutHeight = 0;
    uint32_t alignOutWidth = 0;
    uint32_t padTop = 0;
    uint32_t padBottom = 0;
    uint32_t padLeft = 0;
    uint32_t padRight = 0;
    uint32_t blockNum = 0;
    uint32_t ubFactorElement = 0;
    uint32_t blockIdx = 0;
    uint32_t perBlockCount = 0;
    uint64_t workspacePerCore = 0;
    int64_t batchStride = 0;
    int64_t outBatchStride = 0;
    uint32_t loopNC = 0;
    int64_t ncOffset = 0;
    event_t MTE3ToMTE2Event;

    GlobalTensor<T> mGmX;
    GlobalTensor<T> mGmY;
    GlobalTensor<T> mGmWorkspace;
};

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::Init(const PadV3GradReplicateTilingData &__restrict tilingData,
                                                    GM_ADDR x, GM_ADDR padding, GM_ADDR y, GM_ADDR workspace) {
    batch = tilingData.batch;
    ncPerCore = tilingData.ncPerCore;
    tailNC = tilingData.tailNC;
    height = tilingData.height;
    width = tilingData.width;
    outHeight = tilingData.outHeight;
    outWidth = tilingData.outWidth;
    alignHeight = tilingData.alignHeight;
    alignWidth = tilingData.alignWidth;
    alignOutHeight = tilingData.alignOutHeight;
    alignOutWidth = tilingData.alignOutWidth;
    padTop = tilingData.padTop;
    padBottom = tilingData.padBottom;
    padLeft = tilingData.padLeft;
    padRight = tilingData.padRight;
    blockNum = tilingData.blockNum;
    ubFactorElement = tilingData.ubFactorElement;
    workspacePerCore = tilingData.workspacePerCore / sizeof(T);

    batchStride = height * width;
    outBatchStride = outHeight * outWidth;
    blockIdx = GetBlockIdx();
    perBlockCount = BLOCK_BYTES / sizeof(T);
    MTE3ToMTE2Event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    
    if (blockIdx < tailNC) {
        loopNC = ncPerCore + 1;
        ncOffset = blockIdx * loopNC;
    } else {
        loopNC = ncPerCore;
        ncOffset = blockIdx * ncPerCore + tailNC;
    }

    mGmX.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
    mGmY.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));
    mGmWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace));
}

// init used buffer
template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::InitBuffer(TPipe *inputPipe) {
    pipe = inputPipe;
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        pipe->InitBuffer(xInQueue, BUFFER_NUM, ubFactorElement * sizeof(T) * COPY_ROWS_AND_COLS);
        pipe->InitBuffer(yOutQueue, BUFFER_NUM, ubFactorElement * sizeof(T) * COPY_ROWS_AND_COLS);
        pipe->InitBuffer(transposeBuf, ubFactorElement * sizeof(float) * COPY_ROWS_AND_COLS);
        pipe->InitBuffer(floatCastResBuf, ubFactorElement * sizeof(float) * COPY_ROWS_AND_COLS);
    } else {
        pipe->InitBuffer(xInQueue, BUFFER_NUM, ubFactorElement * sizeof(T) * COPY_ROWS_AND_COLS);
        pipe->InitBuffer(yOutQueue, BUFFER_NUM, ubFactorElement * sizeof(T) * COPY_ROWS_AND_COLS);
    }
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyGm2UB(const int32_t cycleIdx, const int64_t copyCount, 
                                                const int32_t batchIdx, const int64_t ncOffset, const int32_t flag) {
    int64_t gmXOffset;
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(copyCount, perBlockCount) - copyCount), (T)0};
    if (flag == 0) {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS; i++) {
            gmXOffset = i * width + cycleIdx * ubFactorElement + batchIdx * batchStride + ncOffset * batchStride;
            DataCopyPad(xLocal[i * ubFactorElement], mGmX[gmXOffset], copyParams, padParams);
        }
    } else {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS; i++) {
            gmXOffset = (height - COPY_ROWS_AND_COLS + i) * width + cycleIdx * ubFactorElement + 
                        batchIdx * batchStride + ncOffset * batchStride;
            DataCopyPad(xLocal[i * ubFactorElement], mGmX[gmXOffset], copyParams, padParams);
        }
    }
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyInFromGm(const int32_t copyCount, const int64_t offset) {
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(copyCount, perBlockCount) - copyCount), (T)0};
    DataCopyPad(xLocal, mGmX[offset], copyParams, padParams);
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyInput2OutGm(const int32_t copyCount, const int64_t offset) {
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(mGmY[offset], yLocal, copyParams);
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyOut2Gm(const int32_t batchIdx, const int64_t ncOffset,
                                                     const int32_t cycles, const int32_t transBlkIdx,
                                                     const int32_t flag) {
    int64_t gmYOffset1 = 0;
    int64_t gmYOffset2 = 0;
    int64_t gmYOffset3 = 0;
    int64_t gmYOffset4 = 0;
    DataCopyExtParams leftCopyParams{1, (uint32_t)((COPY_ROWS_AND_COLS - padLeft) * sizeof(T)), 0, 0, 0};
    DataCopyExtParams rightCopyParams{1, (uint32_t)((COPY_ROWS_AND_COLS - padRight) * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    if (flag == 0) {
        if (cycles <= COPY_ROWS_AND_COLS - padBottom) {
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - padBottom; i++) {
                gmYOffset1 = outWidth * (outHeight - (COPY_ROWS_AND_COLS - padBottom) + i) + batchIdx * outBatchStride +
                             ncOffset * outBatchStride;
                DataCopyPad(mGmY[gmYOffset1], yLocal[i * COPY_ROWS_AND_COLS], leftCopyParams);
            }
        } else {
            for (size_t i = 0; i < cycles; i++) {
                gmYOffset3 = outWidth * (i + transBlkIdx * ubFactorElement) + batchIdx * outBatchStride +
                             ncOffset * outBatchStride;
                DataCopyPad(mGmY[gmYOffset3], yLocal[i * COPY_ROWS_AND_COLS], leftCopyParams);
            }
        }
    } else {
        if (cycles <= COPY_ROWS_AND_COLS - padBottom) {
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - padBottom; i++) {
                gmYOffset2 = outWidth * (outHeight - (COPY_ROWS_AND_COLS - padBottom) + 1 + i) -
                             (COPY_ROWS_AND_COLS - padRight) + batchIdx * outBatchStride + ncOffset * outBatchStride;
                DataCopyPad(mGmY[gmYOffset2], yLocal[i * COPY_ROWS_AND_COLS], rightCopyParams);
            }
        } else {
            for (size_t i = 0; i < cycles; i++) {
                gmYOffset4 = outWidth * (i + transBlkIdx * ubFactorElement + 1) - (COPY_ROWS_AND_COLS - padRight) +
                             batchIdx * outBatchStride + ncOffset * outBatchStride;
                DataCopyPad(mGmY[gmYOffset4], yLocal[i * COPY_ROWS_AND_COLS], rightCopyParams);
            }
        }
    }
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyGmAndWorkspace2UB1(const int32_t batchIdx, 
                                                const int64_t copyCount, const int64_t ncOffset, const int32_t flag) {
    DataCopyExtParams copyParams{1, (uint32_t)(COPY_ROWS_AND_COLS * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    int64_t workspaceOffset1;
    int64_t workspaceOffset2;
    int64_t xGmOffset;
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    if (flag == 0) {  // 左侧COPY_ROWS_AND_COLS列搬入到ub
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - padTop; i++) {
            workspaceOffset1 = i * width + blockIdx * workspacePerCore;
            DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], mGmWorkspace[workspaceOffset1], copyParams, padParams);
        }
        for (size_t i = COPY_ROWS_AND_COLS; i < height - COPY_ROWS_AND_COLS; i++) {
            xGmOffset = i * width + batchIdx * batchStride + ncOffset * batchStride;
            DataCopyPad(xLocal[(i - padTop) * COPY_ROWS_AND_COLS], mGmX[xGmOffset], copyParams, padParams);
        }
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - padBottom; i++) {
            workspaceOffset2 = (i + COPY_ROWS_AND_COLS - padTop) * width + blockIdx * workspacePerCore;
            DataCopyPad(xLocal[(outHeight - (COPY_ROWS_AND_COLS - padBottom) + i) * COPY_ROWS_AND_COLS],
                        mGmWorkspace[workspaceOffset2], copyParams, padParams);
        }
    } else {  // 右侧COPY_ROWS_AND_COLS列搬入到ub
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - padTop; i++) {
            workspaceOffset1 = (i + 1) * width - COPY_ROWS_AND_COLS + blockIdx * workspacePerCore;
            DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], mGmWorkspace[workspaceOffset1], copyParams, padParams);
        }
        for (size_t i = COPY_ROWS_AND_COLS; i < height - COPY_ROWS_AND_COLS; i++) {
            xGmOffset = (i + 1) * width - COPY_ROWS_AND_COLS + batchIdx * batchStride + ncOffset * batchStride;
            DataCopyPad(xLocal[(i - padTop) * COPY_ROWS_AND_COLS], mGmX[xGmOffset], copyParams, padParams);
        }
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - padBottom; i++) {
            workspaceOffset2 =
                (i + COPY_ROWS_AND_COLS - padTop + 1) * width - COPY_ROWS_AND_COLS + blockIdx * workspacePerCore;
            DataCopyPad(xLocal[(outHeight - (COPY_ROWS_AND_COLS - padBottom) + i) * COPY_ROWS_AND_COLS],
                        mGmWorkspace[workspaceOffset2], copyParams, padParams);
        }
    }
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyGmAndWorkspace2UB2(
                            const int32_t transBlkIdx, const int32_t transTimes,
                            const int32_t cycles, const int32_t batchIdx,
                            const int64_t ncOffset, const int32_t flag) {
    DataCopyExtParams copyParams{1, (uint32_t)(COPY_ROWS_AND_COLS * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    int64_t workspaceOffset1;
    int64_t workspaceOffset2;
    int64_t workspaceOffset3;
    int64_t workspaceOffset4;
    int64_t workspaceOffset5;
    int64_t workspaceOffset6;
    int64_t xGmOffset1;
    int64_t xGmOffset2;
    int64_t xGmOffset3;
    int64_t xGmOffset4;
    int64_t xGmOffset5;
    int64_t xGmOffset6;
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();

    if (flag == 0) {
        if (transBlkIdx == 0) {
            for (size_t i = 0; i < ubFactorElement; i++) {
                xGmOffset1 = (i + padTop) * width + batchIdx * batchStride + ncOffset * batchStride;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], mGmX[xGmOffset1], copyParams, padParams);
            }
            pipe_barrier(PIPE_MTE2);
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - padTop; i++) {
                workspaceOffset1 = i * width + blockIdx * workspacePerCore;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], mGmWorkspace[workspaceOffset1], copyParams, padParams);
            }

        } else if (transBlkIdx > 0 && transBlkIdx < transTimes - 1) {
            for (size_t i = 0; i < ubFactorElement; i++) {
                xGmOffset2 = (ubFactorElement * transBlkIdx + padTop + i) * width + batchIdx * batchStride +
                             ncOffset * batchStride;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], mGmX[xGmOffset2], copyParams, padParams);
            }
        } else if (transBlkIdx == transTimes - 1) {
            if (cycles <= COPY_ROWS_AND_COLS - padBottom) {
                for (size_t i = 0; i < COPY_ROWS_AND_COLS - padBottom; i++) {
                    workspaceOffset2 = (i + COPY_ROWS_AND_COLS - padTop) * width + blockIdx * workspacePerCore;
                    DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], mGmWorkspace[workspaceOffset2], copyParams, padParams);
                }
            } else {
                for (size_t i = 0; i < cycles; i++) {
                    xGmOffset3 = (i + (transTimes - 1) * ubFactorElement + padTop) * width + batchIdx * batchStride +
                                 ncOffset * batchStride;
                    DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], mGmX[xGmOffset3], copyParams, padParams);
                }
                pipe_barrier(PIPE_MTE2);
                for (size_t i = 0; i < COPY_ROWS_AND_COLS - padBottom; i++) {
                    workspaceOffset3 = (i + COPY_ROWS_AND_COLS - padTop) * width + blockIdx * workspacePerCore;
                    DataCopyPad(xLocal[(cycles - (COPY_ROWS_AND_COLS - padBottom) + i) * COPY_ROWS_AND_COLS],
                                mGmWorkspace[workspaceOffset3], copyParams, padParams);
                }
            }
        }
    } else {
        if (transBlkIdx == 0) {
            for (size_t i = 0; i < ubFactorElement; i++) {
                xGmOffset4 = (i + padTop + 1) * width - 16 + batchIdx * batchStride + ncOffset * batchStride;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], mGmX[xGmOffset4], copyParams, padParams);
            }
            pipe_barrier(PIPE_MTE2);
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - padTop; i++) {
                workspaceOffset4 = (i + 1) * width - 16 + blockIdx * workspacePerCore;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], mGmWorkspace[workspaceOffset4], copyParams, padParams);
            }

        } else if (transBlkIdx > 0 && transBlkIdx < transTimes - 1) {
            for (size_t i = 0; i < ubFactorElement; i++) {
                xGmOffset5 = (ubFactorElement * transBlkIdx + padTop + i + 1) * width - 16 + batchIdx * batchStride +
                             ncOffset * batchStride;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], mGmX[xGmOffset5], copyParams, padParams);
            }
        } else if (transBlkIdx == transTimes - 1) {
            if (cycles <= COPY_ROWS_AND_COLS - padBottom) {
                for (size_t i = 0; i < COPY_ROWS_AND_COLS - padBottom; i++) {
                    workspaceOffset5 = (i + COPY_ROWS_AND_COLS + 1 - padTop) * width - 16 + blockIdx * workspacePerCore;
                    DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], mGmWorkspace[workspaceOffset5], copyParams, padParams);
                }
            } else {
                for (size_t i = 0; i < cycles; i++) {
                    xGmOffset6 = (i + (transTimes - 1) * ubFactorElement + padTop + 1) * width - 16 +
                                 batchIdx * batchStride + ncOffset * batchStride;
                    DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], mGmX[xGmOffset6], copyParams, padParams);
                }
                pipe_barrier(PIPE_MTE2);
                for (size_t i = 0; i < COPY_ROWS_AND_COLS - padBottom; i++) {
                    workspaceOffset6 = (i + COPY_ROWS_AND_COLS + 1 - padTop) * width - 16 + blockIdx * workspacePerCore;
                    DataCopyPad(xLocal[(cycles - (COPY_ROWS_AND_COLS - padBottom) + i) * COPY_ROWS_AND_COLS],
                                mGmWorkspace[workspaceOffset6], copyParams, padParams);
                }
            }
        }
    }
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyOut2Workspace(const int32_t tIdx, const int64_t calCount,
                                                            const int32_t flag) {
    int64_t workspaceOffset;
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    if (flag == 0) {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - padTop; i++) {
            workspaceOffset = i * width + tIdx * ubFactorElement + blockIdx * workspacePerCore;
            DataCopyPad(mGmWorkspace[workspaceOffset], yLocal[i * ubFactorElement], copyParams);
        }
    } else {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - padBottom; i++) {
            workspaceOffset =
                (COPY_ROWS_AND_COLS - padTop + i) * width + tIdx * ubFactorElement + blockIdx * workspacePerCore;
            DataCopyPad(mGmWorkspace[workspaceOffset], yLocal[i * ubFactorElement], copyParams);
        }
    }
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::ImplTransposeAndCompute(const int64_t transCount, 
                                                                            const int32_t flag) {
    uint32_t loopTimes = CeilDiv(transCount, TRANSDATA_BASE_H);
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
    if constexpr (AscendC::IsSameType<T, half>::value) {
        for (int i = 0; i < HALF_BLOCK_NUM; i++) {
            xSrcLocalList0[i] = (uint64_t)(xLocal[COPY_ROWS_AND_COLS * i].GetPhyAddr());
            xDstLocalList0[i] = (uint64_t)(yLocal[COPY_ROWS_AND_COLS * i * loopTimes].GetPhyAddr());
            xSrcLocalList1[i] = (uint64_t)(yLocal[COPY_ROWS_AND_COLS * i * loopTimes].GetPhyAddr());
            xDstLocalList1[i] = (uint64_t)(xLocal[COPY_ROWS_AND_COLS * i].GetPhyAddr());
        }
        transDataParams.repeatTimes = loopTimes;
        transDataParams.srcRepStride = TRANSDATA_BASE_H * COPY_ROWS_AND_COLS * sizeof(T) / DATA_BLOCK_BYTES;
        transDataParams.dstRepStride = 1;
        TransDataTo5HD<T>(xDstLocalList0, xSrcLocalList0, transDataParams);
        if (flag == 0) {
            for (size_t i = 0; i < padLeft; i++) {
                Add(yLocal[padLeft * ubFactorElement], yLocal[i * ubFactorElement],
                    yLocal[padLeft * ubFactorElement], ubFactorElement);
            }
            DataCopy(yLocal, yLocal[padLeft * ubFactorElement],
                     (COPY_ROWS_AND_COLS - padLeft) * ubFactorElement);

        } else {
            for (size_t i = 0; i < padRight; i++) {
                Add(yLocal[(COPY_ROWS_AND_COLS - 1 - padRight) * ubFactorElement],
                    yLocal[(COPY_ROWS_AND_COLS - 1 - i) * ubFactorElement],
                    yLocal[(COPY_ROWS_AND_COLS - 1 - padRight) * ubFactorElement], ubFactorElement);
            }
        }
        transDataParams.srcRepStride = 1;
        transDataParams.dstRepStride = TRANSDATA_BASE_H * COPY_ROWS_AND_COLS * sizeof(T) / DATA_BLOCK_BYTES;
        TransDataTo5HD<T>(xDstLocalList1, xSrcLocalList1, transDataParams);
        DataCopy(yLocal, xLocal, COPY_ROWS_AND_COLS * ubFactorElement);
        xInQueue.FreeTensor(xLocal);
        yOutQueue.EnQue(yLocal);
    } else {
        for (size_t time = 0; time < COPY_ROWS_AND_COLS / FLOAT_BLOCK_NUM; time++) {
            for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
                xSrcLocalList0[i] = (uint64_t)(xLocal[COPY_ROWS_AND_COLS * i + FLOAT_BLOCK_NUM * time].GetPhyAddr());
            }
            for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
                xDstLocalList0[CONST_VALUE_2 * i] =
                    (uint64_t)(yLocal[i * ubFactorElement + FLOAT_BLOCK_NUM * ubFactorElement * time]
                                   .GetPhyAddr());
                xDstLocalList0[CONST_VALUE_2 * i + 1] =
                    (uint64_t)(yLocal[i * ubFactorElement + FLOAT_BLOCK_NUM * ubFactorElement * time +
                                             FLOAT_BLOCK_NUM]
                                   .GetPhyAddr());
            }
            transDataParams.repeatTimes = loopTimes;
            transDataParams.srcRepStride = TRANSDATA_BASE_H * COPY_ROWS_AND_COLS * sizeof(T) / DATA_BLOCK_BYTES;
            transDataParams.dstRepStride = COPY_ROWS_AND_COLS / FLOAT_BLOCK_NUM;
            TransDataTo5HD<T>(xDstLocalList0, xSrcLocalList0, transDataParams);
        }
        if (flag == 0) {
            for (size_t i = 0; i < padLeft; i++) {
                Add(yLocal[padLeft * ubFactorElement], yLocal[i * ubFactorElement],
                    yLocal[padLeft * ubFactorElement], ubFactorElement);
            }
            DataCopy(yLocal, yLocal[padLeft * ubFactorElement],
                     (COPY_ROWS_AND_COLS - padLeft) * ubFactorElement);

        } else {
            for (size_t i = 0; i < padRight; i++) {
                Add(yLocal[(COPY_ROWS_AND_COLS - 1 - padRight) * ubFactorElement],
                    yLocal[(COPY_ROWS_AND_COLS - 1 - i) * ubFactorElement],
                    yLocal[(COPY_ROWS_AND_COLS - 1 - padRight) * ubFactorElement], ubFactorElement);
            }
        }
        for (size_t time = 0; time < ubFactorElement / FLOAT_BLOCK_NUM; time++) {
            for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
                xSrcLocalList1[i] =
                    (uint64_t)(yLocal[ubFactorElement * i + time * FLOAT_BLOCK_NUM].GetPhyAddr());
            }
            for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
                xDstLocalList1[CONST_VALUE_2 * i] =
                    (uint64_t)(xLocal[COPY_ROWS_AND_COLS * i + time * COPY_ROWS_AND_COLS * FLOAT_BLOCK_NUM]
                                   .GetPhyAddr());
                xDstLocalList1[CONST_VALUE_2 * i + 1] =
                    (uint64_t)(xLocal[COPY_ROWS_AND_COLS * i + time * COPY_ROWS_AND_COLS * FLOAT_BLOCK_NUM +
                                      FLOAT_BLOCK_NUM]
                                   .GetPhyAddr());
            }
            transDataParams.repeatTimes = 1;
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
            TransDataTo5HD<T>(xDstLocalList1, xSrcLocalList1, transDataParams);
        }
        DataCopy(yLocal, xLocal, COPY_ROWS_AND_COLS * ubFactorElement);
        xInQueue.FreeTensor(xLocal);
        yOutQueue.EnQue(yLocal);
    }
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::ImplTransposeAndComputeBF16(const int64_t transCount, 
                                                                                 const int32_t flag) {
    uint32_t loopTimes = CeilDiv(transCount, TRANSDATA_BASE_H);
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
    Cast(floatTensor, xLocal, RoundMode::CAST_NONE, ubFactorElement * COPY_ROWS_AND_COLS);
    for (size_t time = 0; time < COPY_ROWS_AND_COLS / FLOAT_BLOCK_NUM; time++) {
        for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
            xSrcLocalList0[i] = (uint64_t)(floatTensor[COPY_ROWS_AND_COLS * i + FLOAT_BLOCK_NUM * time].GetPhyAddr());
        }
        for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
            xDstLocalList0[CONST_VALUE_2 * i] =
                (uint64_t)(transposeData[i * ubFactorElement + FLOAT_BLOCK_NUM * ubFactorElement * time]
                                .GetPhyAddr());
            xDstLocalList0[CONST_VALUE_2 * i + 1] =
                (uint64_t)(transposeData[i * ubFactorElement + FLOAT_BLOCK_NUM * ubFactorElement * time +
                                            FLOAT_BLOCK_NUM]
                                .GetPhyAddr());
        }
        transDataParams.repeatTimes = loopTimes;
        transDataParams.srcRepStride = TRANSDATA_BASE_H * COPY_ROWS_AND_COLS * sizeof(float) / DATA_BLOCK_BYTES;
        transDataParams.dstRepStride = COPY_ROWS_AND_COLS / FLOAT_BLOCK_NUM;
        TransDataTo5HD<float>(xDstLocalList0, xSrcLocalList0, transDataParams);
    }
    if (flag == 0) {
        for (size_t i = 0; i < padLeft; i++) {
            Add(transposeData[padLeft * ubFactorElement], transposeData[i * ubFactorElement],
                transposeData[padLeft * ubFactorElement], ubFactorElement);
        }
        DataCopy(transposeData, transposeData[padLeft * ubFactorElement],
                    (COPY_ROWS_AND_COLS - padLeft) * ubFactorElement);

    } else {
        for (size_t i = 0; i < padRight; i++) {
            Add(transposeData[(COPY_ROWS_AND_COLS - 1 - padRight) * ubFactorElement],
                transposeData[(COPY_ROWS_AND_COLS - 1 - i) * ubFactorElement],
                transposeData[(COPY_ROWS_AND_COLS - 1 - padRight) * ubFactorElement], ubFactorElement);
        }
    }
    for (size_t time = 0; time < ubFactorElement / FLOAT_BLOCK_NUM; time++) {
        for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
            xSrcLocalList1[i] =
                (uint64_t)(transposeData[ubFactorElement * i + time * FLOAT_BLOCK_NUM].GetPhyAddr());
        }
        for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
            xDstLocalList1[CONST_VALUE_2 * i] =
                (uint64_t)(floatTensor[COPY_ROWS_AND_COLS * i + time * COPY_ROWS_AND_COLS * FLOAT_BLOCK_NUM]
                                .GetPhyAddr());
            xDstLocalList1[CONST_VALUE_2 * i + 1] =
                (uint64_t)(floatTensor[COPY_ROWS_AND_COLS * i + time * COPY_ROWS_AND_COLS * FLOAT_BLOCK_NUM +
                                    FLOAT_BLOCK_NUM]
                                .GetPhyAddr());
        }
        transDataParams.repeatTimes = 1;
        transDataParams.srcRepStride = 0;
        transDataParams.dstRepStride = 0;
        TransDataTo5HD<float>(xDstLocalList1, xSrcLocalList1, transDataParams);
    }
    Cast(yLocal, floatTensor, RoundMode::CAST_RINT, ubFactorElement * COPY_ROWS_AND_COLS);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::ComputeHGrad(const int32_t calCount, const int32_t flag) {
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    int64_t workspaceOffset1 = 0;
    int64_t workspaceOffset2 = 0;
    int64_t offset3 = 0;
    int64_t yOffset1 = 0;
    int64_t yOffset2 = 0;
    // compute grad
    if (flag == 0) {
        for (size_t i = 0; i < padTop; i++) {
            Add(xLocal[padTop * ubFactorElement], xLocal[i * ubFactorElement],
                xLocal[padTop * ubFactorElement], calCount);
        }
        DataCopy(yLocal, xLocal[padTop * ubFactorElement], (COPY_ROWS_AND_COLS - padTop) * ubFactorElement);
    } else {
        for (size_t i = 0; i < padBottom; i++) {
            Add(xLocal[(COPY_ROWS_AND_COLS - 1 - padBottom) * ubFactorElement],
                xLocal[(COPY_ROWS_AND_COLS - 1 - i) * ubFactorElement],
                xLocal[(COPY_ROWS_AND_COLS - 1 - padBottom) * ubFactorElement], calCount);
        }
        DataCopy(yLocal, xLocal, (COPY_ROWS_AND_COLS - padBottom) * ubFactorElement);
    }
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::ComputeHGradBF16(const int32_t calCount, const int32_t flag) {
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    int64_t workspaceOffset1 = 0;
    int64_t workspaceOffset2 = 0;
    int64_t offset3 = 0;
    int64_t yOffset1 = 0;
    int64_t yOffset2 = 0;
    Cast(floatTensor, xLocal, RoundMode::CAST_NONE, ubFactorElement * COPY_ROWS_AND_COLS);
    // compute grad
    if (flag == 0) {
        for (size_t i = 0; i < padTop; i++) {  // i表示行数下标index
            Add(floatTensor[padTop * ubFactorElement], floatTensor[i * ubFactorElement],
                floatTensor[padTop * ubFactorElement], calCount);
        }
        DataCopy(floatTensor, floatTensor[padTop * ubFactorElement], (COPY_ROWS_AND_COLS - padTop) * ubFactorElement);
    } else {
        for (size_t i = 0; i < padBottom; i++) {  // i表示行数下标index
            Add(floatTensor[(COPY_ROWS_AND_COLS - 1 - padBottom) * ubFactorElement],
                floatTensor[(COPY_ROWS_AND_COLS - 1 - i) * ubFactorElement],
                floatTensor[(COPY_ROWS_AND_COLS - 1 - padBottom) * ubFactorElement], calCount);
        }
    }
    Cast(yLocal, floatTensor, RoundMode::CAST_RINT, ubFactorElement * COPY_ROWS_AND_COLS);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyIn(const int32_t copyCount, const int64_t workspaceOffset) {
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    DataCopyPad(xLocal, mGmWorkspace[workspaceOffset], copyParams, padParams);
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::Compute(const int32_t copyCount) {
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    uint32_t alignCopyCount = CeilAlign(copyCount, perBlockCount);
    DataCopy(yLocal, xLocal, alignCopyCount);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyOut(const int32_t copyCount, const int64_t offset) {
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(mGmY[offset], yLocal, copyParams);
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::Process() {
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
    int64_t calCount = ubFactorElement;
    int64_t cycleTimes = ubFactorElement;
    uint32_t copyCount1 = COPY_ROWS_AND_COLS * ubFactorElement;
    uint32_t copyCount2 = COPY_ROWS_AND_COLS * ubFactorElement;
    uint32_t copyCount = COPY_ROWS_AND_COLS * ubFactorElement;

    uint32_t copyTimesOneRow = CeilDiv(width, ubFactorElement);
    uint32_t transTimesOneCol = CeilDiv(outHeight, ubFactorElement);
    uint32_t copyMidDataTimes = CeilDiv(width - CONST_VALUE_2 * COPY_ROWS_AND_COLS,
                                        COPY_ROWS_AND_COLS * ubFactorElement);

    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        floatTensor = floatCastResBuf.Get<float>();
        transposeData = transposeBuf.Get<float>();
    }

    for (size_t loop = 0; loop < loopNC; loop++) {
        calCount = ubFactorElement;
        for (size_t time = 0; time < copyTimesOneRow; time++) {
            if (time == copyTimesOneRow - 1) {
                calCount = width - (copyTimesOneRow - 1) * ubFactorElement;
            }
            // 搬padTop16行
            CopyGm2UB(time, calCount, loop, ncOffset, 0);
            // padTop累加
            if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                ComputeHGradBF16(calCount, 0);
            } else {
                ComputeHGrad(calCount, 0);
            }
            // 累加轴搬到workspace上
            CopyOut2Workspace(time, calCount, 0);
            // 搬padBottom16行
            CopyGm2UB(time, calCount, loop, ncOffset, 1);
            // padBottom累加
            if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                ComputeHGradBF16(calCount, 1);
            } else {
                ComputeHGrad(calCount, 1);
            }
            // 累加轴搬到workspace上
            CopyOut2Workspace(time, calCount, 1);
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, MTE3ToMTE2Event);
        wait_flag(PIPE_MTE3, PIPE_MTE2, MTE3ToMTE2Event);
        // workspace上padTop方向搬出到gm
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - padTop; i++) {
            copyCount1 = COPY_ROWS_AND_COLS * ubFactorElement;
            for (size_t j = 0; j < copyMidDataTimes; j++) {
                if (j == copyMidDataTimes - 1) {
                    copyCount1 =
                        width - CONST_VALUE_2 * COPY_ROWS_AND_COLS -
                        (copyMidDataTimes - 1) * ubFactorElement * COPY_ROWS_AND_COLS;
                }
                workspaceOffset1 = COPY_ROWS_AND_COLS + j * ubFactorElement * COPY_ROWS_AND_COLS + i * width +
                                   blockIdx * workspacePerCore;
                gmYOffset1 = COPY_ROWS_AND_COLS - padLeft + j * ubFactorElement * COPY_ROWS_AND_COLS + i * outWidth +
                             loop * outBatchStride + ncOffset * outBatchStride;
                CopyIn(copyCount1, workspaceOffset1);
                Compute(ubFactorElement * COPY_ROWS_AND_COLS);
                CopyOut(copyCount1, gmYOffset1);
            }
        }
        // workspace上padBottom方向搬出到gm
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - padBottom; i++) {
            copyCount2 = COPY_ROWS_AND_COLS * ubFactorElement;
            for (size_t j = 0; j < copyMidDataTimes; j++) {
                if (j == copyMidDataTimes - 1) {
                    copyCount2 =
                        width - CONST_VALUE_2 * COPY_ROWS_AND_COLS -
                        (copyMidDataTimes - 1) * ubFactorElement * COPY_ROWS_AND_COLS;
                }
                workspaceOffset2 = COPY_ROWS_AND_COLS + j * ubFactorElement * COPY_ROWS_AND_COLS +
                                   (i + COPY_ROWS_AND_COLS - padTop) * width + blockIdx * workspacePerCore;
                gmYOffset2 = COPY_ROWS_AND_COLS - padLeft + (outHeight - (COPY_ROWS_AND_COLS - padBottom) + i) 
                             * outWidth + j * ubFactorElement * COPY_ROWS_AND_COLS + loop * outBatchStride +
                             ncOffset * outBatchStride;
                CopyIn(copyCount2, workspaceOffset2);
                Compute(copyCount);
                CopyOut(copyCount2, gmYOffset2);
            }
        }
        if (transTimesOneCol == 1) {
            CopyGmAndWorkspace2UB1(loop, COPY_ROWS_AND_COLS, ncOffset, 0);
            if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                ImplTransposeAndComputeBF16(ubFactorElement, 0);
            } else {
                ImplTransposeAndCompute(ubFactorElement, 0);
            }
            CopyOut2Gm(loop, ncOffset, outHeight, 0, 0);
            CopyGmAndWorkspace2UB1(loop, COPY_ROWS_AND_COLS, ncOffset, 1);
            if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                ImplTransposeAndComputeBF16(ubFactorElement, 1);
            } else {
                ImplTransposeAndCompute(ubFactorElement, 1);
            }
            CopyOut2Gm(loop, ncOffset, outHeight, 0, 1);
        } else if (transTimesOneCol > 1) {
            for (size_t transBlk = 0; transBlk < transTimesOneCol; transBlk++) {
                cycleTimes = ubFactorElement;
                if (transBlk == transTimesOneCol - 1) {
                    cycleTimes = outHeight - (transTimesOneCol - 1) * ubFactorElement;
                }
                CopyGmAndWorkspace2UB2(transBlk, transTimesOneCol, cycleTimes, loop, ncOffset, 0);
                if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                    ImplTransposeAndComputeBF16(ubFactorElement, 0);
                } else {
                    ImplTransposeAndCompute(ubFactorElement, 0);
                }
                CopyOut2Gm(loop, ncOffset, cycleTimes, transBlk, 0);
                CopyGmAndWorkspace2UB2(transBlk, transTimesOneCol, cycleTimes, loop, ncOffset, 1);
                if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                    ImplTransposeAndComputeBF16(ubFactorElement, 1);
                } else {
                    ImplTransposeAndCompute(ubFactorElement, 1);
                }
                CopyOut2Gm(loop, ncOffset, cycleTimes, transBlk, 1);
            }
        }
        for (size_t rowIdx = COPY_ROWS_AND_COLS; rowIdx < height - COPY_ROWS_AND_COLS; rowIdx++) {
            copyCount = ubFactorElement * COPY_ROWS_AND_COLS;
            for (size_t i = 0; i < copyMidDataTimes; i++) {
                if (i == copyMidDataTimes - 1) {
                    copyCount =
                        width - CONST_VALUE_2 * COPY_ROWS_AND_COLS -
                        (copyMidDataTimes - 1) * ubFactorElement * COPY_ROWS_AND_COLS;
                }
                gmXOffset1 = COPY_ROWS_AND_COLS + rowIdx * width + i * ubFactorElement * COPY_ROWS_AND_COLS +
                             loop * batchStride + ncOffset * batchStride;
                gmYOffset3 = (COPY_ROWS_AND_COLS - padLeft) + (rowIdx - padTop) * outWidth +
                             i * ubFactorElement * COPY_ROWS_AND_COLS + loop * outBatchStride +
                             ncOffset * outBatchStride;
                CopyInFromGm(copyCount, gmXOffset1);
                Compute(copyCount);
                CopyInput2OutGm(copyCount, gmYOffset3);
            }
        }
    }
}
#endif  // _PAD_V3_GRAD_REPLICATE_H_W_LARGE_