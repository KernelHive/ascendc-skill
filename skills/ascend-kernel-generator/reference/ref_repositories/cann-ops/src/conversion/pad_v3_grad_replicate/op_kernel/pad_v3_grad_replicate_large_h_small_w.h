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
 * \file pad_v3_grad_replicate_large_h_small_w.h
 * \brief
 */
#ifndef _PAD_V3_GRAD_REPLICATE_LARGE_H_SMALL_W_H_
#define _PAD_V3_GRAD_REPLICATE_LARGE_H_SMALL_W_H_

#include "pad_v3_grad_replicate_base.h"

template <typename T>
class PadV3GradReplicateLargeHSmallW {
public:
    __aicore__ inline PadV3GradReplicateLargeHSmallW() {};
    __aicore__ inline void Init(const PadV3GradReplicateTilingData &__restrict tilingData, 
                                GM_ADDR x, GM_ADDR padding, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void InitBuffer(TPipe *inputPipe);
    __aicore__ inline void CopyGm2UB(const int32_t batchIdx, const int32_t flag);
    __aicore__ inline void CopyOut2Gm(const int32_t batchIdx, const int32_t cycles, const int32_t transBlkIdx);
    __aicore__ inline void CopyGmAndWs2UB1(const int32_t batchIdx);
    __aicore__ inline void CopyGmAndWorkspace2UB2(const int32_t transBlkIdx, const int32_t transTimes,
                                                  const int32_t cycles, const int32_t batchIdx);
    __aicore__ inline void CopyOut2Ws(const int64_t calCount, const int32_t flag);
    __aicore__ inline void ComputeHGrad(const int32_t calCount, const int32_t flag);
    __aicore__ inline void ImplTransposeAndCompute(const int64_t transCount);
    __aicore__ inline void Process();

private:
    TPipe *pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, 1> xInQueue;
    TQue<QuePosition::VECOUT, 1> yOutQueue;
    TQue<QuePosition::VECOUT, 1> transposeQue;

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
__aicore__ inline void PadV3GradReplicateLargeHSmallW<T>::Init(
                                                    const PadV3GradReplicateTilingData &__restrict tilingData,
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
__aicore__ inline void PadV3GradReplicateLargeHSmallW<T>::InitBuffer(TPipe *inputPipe) {
    pipe = inputPipe;
    pipe->InitBuffer(xInQueue, 1, ubFactorElement * SMALL_WIDTH_LIMIT * sizeof(T));
    pipe->InitBuffer(yOutQueue, 1, ubFactorElement * SMALL_WIDTH_LIMIT * sizeof(T));
    pipe->InitBuffer(transposeQue, 1, ubFactorElement * SMALL_WIDTH_LIMIT * sizeof(T));
}

template <typename T>
__aicore__ inline void PadV3GradReplicateLargeHSmallW<T>::CopyGm2UB(const int32_t batchIdx, const int32_t flag) {
    int64_t gmXOffset;
    int32_t alignCopyCount = CeilAlign(width, perBlockCount);
    DataCopyExtParams copyParams{1, (uint32_t)(width * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {true, 0, (uint8_t)(alignCopyCount - width), (T)0};
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    if (flag == 0) {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS; i++) {
            gmXOffset = i * width + batchIdx * batchStride + ncOffset * batchStride;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], mGmX[gmXOffset], copyParams, padParams);
        }
    } else {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS; i++) {
            gmXOffset = (height - COPY_ROWS_AND_COLS + i) * width + batchIdx * batchStride + ncOffset * batchStride;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], mGmX[gmXOffset], copyParams, padParams);
        }
    }
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateLargeHSmallW<T>::CopyOut2Gm(const int32_t batchIdx, const int32_t cycles,
                                                                     const int32_t transBlkIdx) {
    int64_t gmYOffset1 = 0;
    int64_t gmYOffset2 = 0;
    DataCopyExtParams copyParams{1, (uint32_t)(outWidth * sizeof(T)), 0, 0, 0};
    LocalTensor<T> transposeData = transposeQue.DeQue<T>();
    if (cycles <= COPY_ROWS_AND_COLS - padBottom) {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - padBottom; i++) {
            gmYOffset1 = outWidth * (outHeight - (COPY_ROWS_AND_COLS - padBottom) + i) +
                         batchIdx * outBatchStride + ncOffset * outBatchStride;
            DataCopyPad(mGmY[gmYOffset1], transposeData[i * SMALL_WIDTH_LIMIT], copyParams);
        }
    } else {
        for (size_t i = 0; i < cycles; i++) {
            gmYOffset2 = outWidth * (i + transBlkIdx * ubFactorElement) + batchIdx * outBatchStride +
                         ncOffset * outBatchStride;
            DataCopyPad(mGmY[gmYOffset2], transposeData[i * SMALL_WIDTH_LIMIT], copyParams);
        }
    }
    transposeQue.FreeTensor(transposeData);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateLargeHSmallW<T>::CopyGmAndWs2UB1(const int32_t batchIdx) {
    DataCopyExtParams copyParams{1, (uint32_t)(width * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    int64_t workspaceOffset1;
    int64_t workspaceOffset2;
    int64_t xGmOffset;
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    for (size_t i = 0; i < COPY_ROWS_AND_COLS - padTop; i++) {
        workspaceOffset1 = i * width + blockIdx * workspacePerCore;
        DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], mGmWorkspace[workspaceOffset1], copyParams, padParams);
    }
    for (size_t i = COPY_ROWS_AND_COLS; i < height - COPY_ROWS_AND_COLS; i++) {
        xGmOffset = i * width + batchIdx * batchStride + ncOffset * batchStride;
        DataCopyPad(xLocal[(i - padTop) * SMALL_WIDTH_LIMIT], mGmX[xGmOffset], copyParams, padParams);
    }
    for (size_t i = 0; i < COPY_ROWS_AND_COLS - padBottom; i++) {
        workspaceOffset2 = (i + COPY_ROWS_AND_COLS - padTop) * width + blockIdx * workspacePerCore;
        DataCopyPad(xLocal[(outHeight - (COPY_ROWS_AND_COLS - padBottom) + i) * SMALL_WIDTH_LIMIT],
                    mGmWorkspace[workspaceOffset2], copyParams, padParams);
    }
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateLargeHSmallW<T>::CopyGmAndWorkspace2UB2(const int32_t transBlkIdx,
                                                                        const int32_t transTimes, const int32_t cycles,
                                                                        const int32_t batchIdx) {
    DataCopyExtParams copyParams{1, (uint32_t)(width * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    int64_t workspaceOffset1;
    int64_t workspaceOffset2;
    int64_t workspaceOffset3;
    int64_t xGmOffset1;
    int64_t xGmOffset2;
    int64_t xGmOffset3;
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();

    if (transBlkIdx == 0) {
        for (size_t i = 0; i < ubFactorElement; i++) {
            xGmOffset1 = (i + padTop) * width + batchIdx * batchStride + ncOffset * batchStride;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], mGmX[xGmOffset1], copyParams, padParams);
        }
        pipe_barrier(PIPE_MTE2);
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - padTop; i++) {
            workspaceOffset1 = i * width + blockIdx * workspacePerCore;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], mGmWorkspace[workspaceOffset1], copyParams, padParams);
        }

    } else if (transBlkIdx > 0 && transBlkIdx < transTimes - 1) {
        for (size_t i = 0; i < ubFactorElement; i++) {
            xGmOffset2 = (ubFactorElement * transBlkIdx + padTop + i) * width +
                         batchIdx * batchStride + ncOffset * batchStride;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], mGmX[xGmOffset2], copyParams, padParams);
        }
    } else if (transBlkIdx == transTimes - 1) {
        if (cycles <= COPY_ROWS_AND_COLS - padBottom) {
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - padBottom; i++) {
                workspaceOffset2 =
                    (i + COPY_ROWS_AND_COLS - padTop) * width + blockIdx * workspacePerCore;
                DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], mGmWorkspace[workspaceOffset2], copyParams,
                            padParams);
            }
        } else {
            for (size_t i = 0; i < cycles; i++) {
                xGmOffset3 = (i + (transTimes - 1) * ubFactorElement + padTop) * width +
                             batchIdx * batchStride + ncOffset * batchStride;
                DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], mGmX[xGmOffset3], copyParams, padParams);
            }
            pipe_barrier(PIPE_MTE2);
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - padBottom; i++) {
                workspaceOffset3 = (i + COPY_ROWS_AND_COLS - padTop) * width + blockIdx * workspacePerCore;
                DataCopyPad(xLocal[(cycles - (COPY_ROWS_AND_COLS - padBottom) + i) * SMALL_WIDTH_LIMIT],
                            mGmWorkspace[workspaceOffset3], copyParams, padParams);
            }
        }
    }
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateLargeHSmallW<T>::CopyOut2Ws(const int64_t calCount, const int32_t flag) {
    int64_t workspaceOffset;
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    if (flag == 0) {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - padTop; i++) {
            workspaceOffset = i * width + blockIdx * workspacePerCore;
            DataCopyPad(mGmWorkspace[workspaceOffset], yLocal[i * SMALL_WIDTH_LIMIT], copyParams);
        }
    } else {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - padBottom; i++) {
            workspaceOffset = (COPY_ROWS_AND_COLS - padTop + i) * width + blockIdx * workspacePerCore;
            DataCopyPad(mGmWorkspace[workspaceOffset], yLocal[i * SMALL_WIDTH_LIMIT], copyParams);
        }
    }
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateLargeHSmallW<T>::ImplTransposeAndCompute(const int64_t transCount) {
    uint32_t loopTimes = CeilDiv(transCount, TRANSDATA_BASE_H);
    uint64_t xSrcLocalList0[TRANSDATA_BASE_H];
    uint64_t xDstLocalList0[TRANSDATA_BASE_H];
    uint64_t xSrcLocalList1[TRANSDATA_BASE_H];
    uint64_t xDstLocalList1[TRANSDATA_BASE_H];
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
                    (uint64_t)(transposeData[ubFactorElement * i + time * ubFactorElement * TRANSDATA_BASE_H]
                    .GetPhyAddr());
            }
            transDataParams.repeatTimes = loopTimes;
            transDataParams.srcRepStride = TRANSDATA_BASE_H * SMALL_WIDTH_LIMIT * sizeof(T) / DATA_BLOCK_BYTES;
            transDataParams.dstRepStride = 1;
            TransDataTo5HD<T>(xDstLocalList0, xSrcLocalList0, transDataParams);
        }

        for (size_t i = 0; i < padLeft; i++) {
            Add(transposeData[padLeft * ubFactorElement], transposeData[i * ubFactorElement],
                transposeData[padLeft * ubFactorElement], ubFactorElement);
        }
        for (size_t i = 0; i < padRight; i++) {
            Add(transposeData[(width - 1 - padRight) * ubFactorElement],
                transposeData[(width - 1 - i) * ubFactorElement],
                transposeData[(width - 1 - padRight) * ubFactorElement], ubFactorElement);
        }
        DataCopy(transposeData, transposeData[padLeft * ubFactorElement], outWidth * ubFactorElement);
        for (size_t time = 0; time < loopTimes; time++) {
            for (int i = 0; i < HALF_BLOCK_NUM; i++) {
                xSrcLocalList1[i] =
                    (uint64_t)(transposeData[ubFactorElement * i + time * HALF_BLOCK_NUM].GetPhyAddr());
                xDstLocalList1[i] =
                    (uint64_t)(xLocal[SMALL_WIDTH_LIMIT * i + time * HALF_BLOCK_NUM * SMALL_WIDTH_LIMIT].GetPhyAddr());
            }
            transDataParams.repeatTimes = SMALL_WIDTH_LIMIT / TRANSDATA_BASE_H;
            transDataParams.srcRepStride = ubFactorElement;
            transDataParams.dstRepStride = 1;
            TransDataTo5HD<T>(xDstLocalList1, xSrcLocalList1, transDataParams);
        }
        DataCopy(transposeData, xLocal, SMALL_WIDTH_LIMIT * ubFactorElement);
        xInQueue.FreeTensor(xLocal);
        transposeQue.EnQue(transposeData);
    } else {
        for (size_t time = 0; time < SMALL_WIDTH_LIMIT / FLOAT_BLOCK_NUM; time++) {
            for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
                xSrcLocalList0[i] = (uint64_t)(xLocal[SMALL_WIDTH_LIMIT * i + FLOAT_BLOCK_NUM * time].GetPhyAddr());
            }
            for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
                xDstLocalList0[CONST_VALUE_2 * i] = (uint64_t)(transposeData[i * ubFactorElement +
                                                                 FLOAT_BLOCK_NUM * ubFactorElement * time]
                                                       .GetPhyAddr());
                xDstLocalList0[CONST_VALUE_2 * i + 1] =
                    (uint64_t)(transposeData[i * ubFactorElement +
                                             FLOAT_BLOCK_NUM * ubFactorElement * time + FLOAT_BLOCK_NUM]
                                   .GetPhyAddr());
            }
            transDataParams.repeatTimes = loopTimes;
            transDataParams.srcRepStride = TRANSDATA_BASE_H * SMALL_WIDTH_LIMIT * sizeof(T) / DATA_BLOCK_BYTES;
            transDataParams.dstRepStride = TRANSDATA_BASE_H / FLOAT_BLOCK_NUM;
            TransDataTo5HD<T>(xDstLocalList0, xSrcLocalList0, transDataParams);
        }
        for (size_t i = 0; i < padLeft; i++) {
            Add(transposeData[padLeft * ubFactorElement], transposeData[i * ubFactorElement],
                transposeData[padLeft * ubFactorElement], ubFactorElement);
        }
        for (size_t i = 0; i < padRight; i++) {
            Add(transposeData[(width - 1 - padRight) * ubFactorElement],
                transposeData[(width - 1 - i) * ubFactorElement],
                transposeData[(width - 1 - padRight) * ubFactorElement], ubFactorElement);
        }
        DataCopy(transposeData, transposeData[padLeft * ubFactorElement], outWidth * ubFactorElement);

        for (size_t time = 0; time < ubFactorElement / FLOAT_BLOCK_NUM; time++) {
            for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
                xSrcLocalList1[i] =
                    (uint64_t)(transposeData[ubFactorElement * i + time * FLOAT_BLOCK_NUM].GetPhyAddr());
            }
            for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
                xDstLocalList1[CONST_VALUE_2 * i] =
                    (uint64_t)(xLocal[SMALL_WIDTH_LIMIT * i + time * SMALL_WIDTH_LIMIT * FLOAT_BLOCK_NUM]
                                   .GetPhyAddr());
                xDstLocalList1[CONST_VALUE_2 * i + 1] =
                    (uint64_t)(xLocal[SMALL_WIDTH_LIMIT * i + time * SMALL_WIDTH_LIMIT * FLOAT_BLOCK_NUM +
                                      FLOAT_BLOCK_NUM]
                                   .GetPhyAddr());
            }
            transDataParams.repeatTimes = SMALL_WIDTH_LIMIT / TRANSDATA_BASE_H;
            transDataParams.srcRepStride = CONST_VALUE_2 * ubFactorElement;
            transDataParams.dstRepStride = TRANSDATA_BASE_H / FLOAT_BLOCK_NUM;
            TransDataTo5HD<T>(xDstLocalList1, xSrcLocalList1, transDataParams);
        }
        DataCopy(transposeData, xLocal, SMALL_WIDTH_LIMIT * ubFactorElement);
        xInQueue.FreeTensor(xLocal);
        transposeQue.EnQue(transposeData);
    }
}

template <typename T>
__aicore__ inline void PadV3GradReplicateLargeHSmallW<T>::ComputeHGrad(const int32_t calCount, const int32_t flag) {
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    // compute grad
    if (flag == 0) {
        for (size_t i = 0; i < padTop; i++) {
            Add(xLocal[padTop * SMALL_WIDTH_LIMIT], xLocal[i * SMALL_WIDTH_LIMIT],
                xLocal[padTop * SMALL_WIDTH_LIMIT], calCount);
        }
        DataCopy(yLocal, xLocal[padTop * SMALL_WIDTH_LIMIT],
                 (COPY_ROWS_AND_COLS - padTop) * SMALL_WIDTH_LIMIT);
    } else {
        for (size_t i = 0; i < padBottom; i++) {
            Add(xLocal[(COPY_ROWS_AND_COLS - 1 - padBottom) * SMALL_WIDTH_LIMIT],
                xLocal[(COPY_ROWS_AND_COLS - 1 - i) * SMALL_WIDTH_LIMIT],
                xLocal[(COPY_ROWS_AND_COLS - 1 - padBottom) * SMALL_WIDTH_LIMIT], calCount);
        }
        DataCopy(yLocal, xLocal, (COPY_ROWS_AND_COLS - padBottom) * SMALL_WIDTH_LIMIT);
    }
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateLargeHSmallW<T>::Process() {
    int64_t calCount = width;
    int64_t cycleTimes = ubFactorElement;

    uint32_t transTimesOneCol = CeilDiv(outHeight, ubFactorElement);

    for (size_t loop = 0; loop < loopNC; loop++) {
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
            ImplTransposeAndCompute(ubFactorElement);
            CopyOut2Gm(loop, outHeight, 0);
        } else if (transTimesOneCol > 1) {
            for (size_t transBlk = 0; transBlk < transTimesOneCol; transBlk++) {
                cycleTimes = ubFactorElement;
                if (transBlk == transTimesOneCol - 1) {
                    cycleTimes = outHeight - (transTimesOneCol - 1) * ubFactorElement;
                }
                CopyGmAndWorkspace2UB2(transBlk, transTimesOneCol, cycleTimes, loop);
                ImplTransposeAndCompute(ubFactorElement);
                CopyOut2Gm(loop, cycleTimes, transBlk);
            }
        }
    }
}
#endif  // _PAD_V3_GRAD_REPLICATE_LARGE_H_SMALL_W_H_