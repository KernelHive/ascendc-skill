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
 * \file pad_v3_grad_replicate_h_w.h
 * \brief
 */
#ifndef _PAD_V3_GRAD_REPLICATE_H_W_
#define _PAD_V3_GRAD_REPLICATE_H_W_

#include "kernel_operator.h"
#include "pad_v3_grad_replicate_base.h"

using namespace AscendC;

template <typename T>
class PadV3GradReplicateHW {
public:
    __aicore__ inline PadV3GradReplicateHW() {};
    __aicore__ inline void Init(const PadV3GradReplicateTilingData &__restrict tilingData, 
                                GM_ADDR x, GM_ADDR padding, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void InitBuffer(TPipe *inputPipe);
    __aicore__ inline void CopyGm2UBWhole(const int64_t offset, const int64_t copyCount);
    __aicore__ inline void CopyGm2UB(const int64_t offset1, const int64_t offset2, const int64_t copyCount);
    __aicore__ inline void CopyWorkspace2Out(const int64_t offset1, const int64_t offset2, const int64_t copyCount);
    __aicore__ inline void CopyOut2Workspace(const int64_t offset, const int64_t calCount);
    __aicore__ inline void CopyOut2Gm(const int64_t offset, const int64_t calCount);
    __aicore__ inline void ComputeHGrad(const int64_t calCount);
    __aicore__ inline void ComputeHGradBF16(const int64_t calCount);
    __aicore__ inline void ComputeDiagonalGrad(const int64_t offset, const int64_t calCount);
    __aicore__ inline void ComputeDiagonalGradBF16(const int64_t offset, const int64_t calCount);
    __aicore__ inline void ComputeWGrad(const int64_t calCount);
    __aicore__ inline void ComputeWGradBF16(const int64_t calCount);
    __aicore__ inline void FloatCast2BF16(const int64_t calCount);
    __aicore__ inline void Process();

private:
    TPipe *pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yOutQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> floatQueue;
    TBuf<TPosition::VECCALC> floatCastResBuf;
    LocalTensor<float> floatTensor;

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
    uint32_t batchOffset = 0;
    uint32_t blockIdx = 0;
    uint32_t perBlockCount = 0;
    int64_t baseGradGmOffset = 0;
    int64_t gradGmOffset = 0;
    int64_t baseGmOffset = 0;
    int64_t xGmOffset = 0;
    int64_t batchStride = 0;
    int64_t outBatchStride = 0;
    event_t eventId0;
    event_t eventId1;
    event_t eventId2;

    GlobalTensor<T> mGmX;
    GlobalTensor<T> mGmY;
    GlobalTensor<T> mGmWorkspace;
};

template <typename T>
__aicore__ inline void PadV3GradReplicateHW<T>::Init(const PadV3GradReplicateTilingData &__restrict tilingData, 
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

    batchStride = height * width;
    outBatchStride = outHeight * outWidth;
    blockIdx = GetBlockIdx();
    perBlockCount = BLOCK_BYTES / sizeof(T);
    eventId0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    eventId1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    eventId2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));

    mGmX.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
    mGmY.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));
    mGmWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace));
}

// init used buffer
template <typename T>
__aicore__ inline void PadV3GradReplicateHW<T>::InitBuffer(TPipe *inputPipe) {
    pipe = inputPipe;
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        pipe->InitBuffer(xInQueue, BUFFER_NUM, ubFactorElement * sizeof(T) * COPY_ROWS_AND_COLS);
        pipe->InitBuffer(yOutQueue, BUFFER_NUM, ubFactorElement * sizeof(T) * COPY_ROWS_AND_COLS);
        pipe->InitBuffer(floatQueue, BUFFER_NUM, ubFactorElement * sizeof(float));
        pipe->InitBuffer(floatCastResBuf, ubFactorElement * sizeof(float));
    } else {
        pipe->InitBuffer(xInQueue, BUFFER_NUM, ubFactorElement * sizeof(T) * COPY_ROWS_AND_COLS);
        pipe->InitBuffer(yOutQueue, BUFFER_NUM, ubFactorElement * sizeof(T) * COPY_ROWS_AND_COLS);
    }
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHW<T>::CopyGm2UBWhole(const int64_t offset, const int64_t copyCount) {
    LocalTensor<T> dataLocal = xInQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(copyCount, perBlockCount) - copyCount), (T)0};
    DataCopyPad(dataLocal[0], mGmX[offset], copyParams, padParams);
    xInQueue.EnQue(dataLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHW<T>::CopyGm2UB(const int64_t offset1, const int64_t offset2, 
                                                          const int64_t copyCount) {
    LocalTensor<T> dataLocal = xInQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(copyCount, perBlockCount) - copyCount), (T)0};
    DataCopyPad(dataLocal[0], mGmX[offset1], copyParams, padParams);
    pipe_barrier(PIPE_MTE2);
    DataCopyPad(dataLocal[copyCount], mGmX[offset2], copyParams, padParams);
    xInQueue.EnQue(dataLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHW<T>::CopyWorkspace2Out(const int64_t offset1, const int64_t offset2, 
                                                                  const int64_t copyCount) {
    LocalTensor<T> dataLocal = yOutQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(copyCount, perBlockCount) - copyCount), (T)0};
    wait_flag(PIPE_S, PIPE_MTE2, eventId0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, eventId1);
    DataCopyPad(dataLocal, mGmWorkspace[offset1], copyParams, padParams);
    set_flag(PIPE_S, PIPE_MTE2, eventId0);
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    set_flag(PIPE_MTE2, PIPE_MTE3, eventID);
    wait_flag(PIPE_MTE2, PIPE_MTE3, eventID);
    wait_flag(PIPE_S, PIPE_MTE3, eventId2);
    DataCopyPad(mGmY[offset2], dataLocal, copyParams);
    set_flag(PIPE_S, PIPE_MTE3, eventId2);
    set_flag(PIPE_MTE3, PIPE_MTE2, eventId1);
    yOutQueue.FreeTensor(dataLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHW<T>::CopyOut2Workspace(const int64_t offset, const int64_t calCount) {
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(mGmWorkspace[offset], yLocal, copyParams);  // 拷贝到workspace，注意计算偏移
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHW<T>::CopyOut2Gm(const int64_t offset, const int64_t calCount) {
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(mGmY[offset], yLocal, copyParams);
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHW<T>::ComputeHGrad(const int64_t calCount) {
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal;
    if (yOutQueue.HasTensorInQue()) {
        yLocal = yOutQueue.DeQue<T>();
    } else {
        yLocal = yOutQueue.AllocTensor<T>();
        T inputValue(0.0);
        Duplicate<T>(yLocal, inputValue, calCount);
    }
    pipe_barrier(PIPE_V);
    Add(yLocal, yLocal, xLocal, calCount);
    yOutQueue.EnQue(yLocal);
    xInQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHW<T>::ComputeHGradBF16(const int64_t calCount) {
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    Cast(floatTensor, xLocal, RoundMode::CAST_NONE, ubFactorElement);
    LocalTensor<float> floatLocal;
    if (floatQueue.HasTensorInQue()) {
        floatLocal = floatQueue.DeQue<float>();
    } else {
        floatLocal = floatQueue.AllocTensor<float>();
        float inputValue(0.0);
        Duplicate<float>(floatLocal, inputValue, calCount);
    }
    pipe_barrier(PIPE_V);
    Add(floatLocal, floatLocal, floatTensor, calCount);
    floatQueue.EnQue(floatLocal);
    xInQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHW<T>::FloatCast2BF16(const int64_t calCount) {
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    LocalTensor<float> floatLocal = floatQueue.DeQue<float>();
    Cast(yLocal, floatLocal, RoundMode::CAST_RINT, calCount);
    yOutQueue.EnQue(yLocal);
    floatQueue.FreeTensor(floatLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHW<T>::ComputeDiagonalGrad(const int64_t offset, const int64_t calCount) {
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    T val1;
    T val2;
    T val3;
    T val4;
    T tmp1;
    T tmp2;
    for (size_t i = 0; i < padLeft; i++) {
        val1 = yLocal.GetValue(i);  // index
        val2 = yLocal.GetValue(padLeft);  // index 边缘轴
        if constexpr(AscendC::IsSameType<T, half>::value) {
            tmp1 = (T)((float)val1 + (float)val2);
            yLocal.SetValue(padLeft, tmp1);
        } else {
            yLocal.SetValue(padLeft, val1 + val2);
        }
    }
    for (size_t i = 0; i < padRight; i++) {
        val3 = yLocal.GetValue(width - 1 - i);  // index
        val4 = yLocal.GetValue(width - 1 - padRight);  // index 边缘轴
        if constexpr(AscendC::IsSameType<T, half>::value) {
            tmp2 = (T)((float)val3 + (float)val4);
            yLocal.SetValue(width - 1 - padRight, tmp2);
        } else {
            yLocal.SetValue(width - 1 - padRight, val3 + val4);
        }
    }
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(mGmWorkspace[offset], yLocal, copyParams);  // 拷贝到workspace，注意计算偏移
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHW<T>::ComputeDiagonalGradBF16(const int64_t offset, const int64_t calCount) {
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    LocalTensor<float> floatLocal = floatQueue.DeQue<float>();
    float val1;
    float val2;
    float val3;
    float val4;
    for (size_t i = 0; i < padLeft; i++) {
        val1 = floatLocal.GetValue(i);  // index
        val2 = floatLocal.GetValue(padLeft);  // index 边缘轴
        floatLocal.SetValue(padLeft, val1 + val2);
    }
    for (size_t i = 0; i < padRight; i++) {
        val3 = floatLocal.GetValue(width - 1 - i);  // index
        val4 = floatLocal.GetValue(width - 1 - padRight);  // index 边缘轴
        floatLocal.SetValue(width - 1 - padRight, val3 + val4);
    }
    Cast(yLocal, floatLocal, RoundMode::CAST_RINT, calCount);
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(mGmWorkspace[offset], yLocal, copyParams);  // 拷贝到workspace，注意计算偏移
    yOutQueue.FreeTensor(yLocal);
    floatQueue.FreeTensor(floatLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHW<T>::ComputeWGrad(const int64_t calCount) {
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    T val1;
    T val2;
    T val3;
    T val4;
    T tmp1;
    T tmp2;
    for (size_t i = 0; i < padLeft; i++) {
        val1 = yLocal.GetValue(i);  // index
        val2 = yLocal.GetValue(padLeft);  // index 边缘轴
        if constexpr(AscendC::IsSameType<T, half>::value) {
            tmp1 = (T)((float)val1 + (float)val2);
            yLocal.SetValue(padLeft, tmp1);
        } else {
            yLocal.SetValue(padLeft, val1 + val2);
        }
    }
    for (size_t i = 0; i < padRight; i++) {
        val3 = yLocal.GetValue(CONST_VALUE_2 * calCount - 1 - i);  // index
        val4 = yLocal.GetValue(CONST_VALUE_2 * calCount - 1 - padRight);  // index 边缘轴
        if constexpr(AscendC::IsSameType<T, half>::value) {
            tmp2 = (T)((float)val3 + (float)val4);
            yLocal.SetValue(CONST_VALUE_2 * calCount - 1 - padRight, tmp2);
        } else {
            yLocal.SetValue(CONST_VALUE_2 * calCount - 1 - padRight, val3 + val4);
        }
    }
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHW<T>::ComputeWGradBF16(const int64_t calCount) {
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    LocalTensor<float> floatLocal = floatQueue.DeQue<float>();
    float val1;
    float val2;
    float val3;
    float val4;
    for (size_t i = 0; i < padLeft; i++) {
        val1 = floatLocal.GetValue(i);  // index
        val2 = floatLocal.GetValue(padLeft);  // index 边缘轴
        floatLocal.SetValue(padLeft, val1 + val2);
    }
    for (size_t i = 0; i < padRight; i++) {
        val3 = floatLocal.GetValue(CONST_VALUE_2 * calCount - 1 - i);  // index
        val4 = floatLocal.GetValue(CONST_VALUE_2 * calCount - 1 - padRight);  // index 边缘轴
        floatLocal.SetValue(CONST_VALUE_2 * calCount - 1 - padRight, val3 + val4);
    }
    Cast(yLocal, floatLocal, RoundMode::CAST_ROUND, CONST_VALUE_2 * calCount);
    floatQueue.FreeTensor(floatLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHW<T>::Process() {
    uint32_t loopNC = 0;
    int64_t ncOffset = 0;
    int64_t gmXOffset;
    int64_t gmXOffset1;
    int64_t gmXOffset2;
    int64_t gmXOffset3;
    int64_t gmYOffset;
    int64_t gmYOffset1;
    int64_t gmYOffset2;
    int64_t gmYOffset3;
    int64_t workspaceOffset;
    int64_t workspaceOffsetOut;
    int64_t workspaceOffset1;
    int64_t workspaceOffset2;
    int64_t workspaceOffset3;

    if (blockIdx < tailNC) {
        loopNC = ncPerCore + 1;
        ncOffset = blockIdx * loopNC;
    } else {
        loopNC = ncPerCore;
        ncOffset = blockIdx * ncPerCore + tailNC;
    }

    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        floatTensor = floatCastResBuf.Get<float>();
    }

    // H方向
    // 对齐场景下，ubFactorElement应为16的倍数，ubFactorElement：一行元素个数
    uint32_t copyTimesOneLine = CeilDiv(width, ubFactorElement);
    for (size_t loop = 0; loop < loopNC; loop++) {
        int64_t calCount = ubFactorElement;
        int64_t copyCount = ubFactorElement;
        // 场景1：若输出shape的H维度为1，padTop和padBottom累加的边缘行重叠，则padTop和padBottom的梯度都要全部累加到outHeight上
        // 子场景1：ub一行能放下，整行搬入处理
        if (outHeight == 1 && copyTimesOneLine == 1) {
            calCount = width;
            copyCount = outWidth;
            for (size_t i = 0; i < height; i++) {
                gmXOffset = i * width + loop * batchStride + ncOffset * batchStride;
                set_flag(PIPE_S, PIPE_MTE2, eventId0);
                wait_flag(PIPE_S, PIPE_MTE2, eventId0);
                CopyGm2UBWhole(gmXOffset, calCount);
                if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                    ComputeHGradBF16(calCount);  // 计算结果存在floatQueue
                } else {
                    ComputeHGrad(calCount);  //计算结果存在yOutQueue
                }
            }
            // workspace上搬入整行
            workspaceOffset = loop * batchStride + ncOffset * batchStride;
            set_flag(PIPE_S, PIPE_MTE3, eventId2);
            wait_flag(PIPE_S, PIPE_MTE3, eventId2);
            // padLeft和padRight的梯度累加到对角线元素上，并将ub整行搬运到workspace上
            if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                ComputeDiagonalGradBF16(workspaceOffset, calCount);
            } else {
                ComputeDiagonalGrad(workspaceOffset, calCount);
            }
            // 计算需要搬出的workspace偏移，需要搬出的起始位置，即左侧边缘行
            workspaceOffsetOut = padLeft + loop * batchStride + ncOffset * batchStride;
            set_flag(PIPE_S, PIPE_MTE2, eventId0);
            // 计算搬出到gm上的偏移，输出到边缘首行，即outWidth左侧的起始位置，index0
            gmYOffset = loop * outBatchStride + ncOffset * outBatchStride;
            set_flag(PIPE_S, PIPE_MTE3, eventId2);
            // workspace -> ub -> gm
            set_flag(PIPE_MTE3, PIPE_MTE2, eventId1);
            CopyWorkspace2Out(workspaceOffsetOut, gmYOffset, copyCount);
            wait_flag(PIPE_S, PIPE_MTE2, eventId0);
            wait_flag(PIPE_S, PIPE_MTE3, eventId2);
            wait_flag(PIPE_MTE3, PIPE_MTE2, eventId1);
        }
        // 子场景2：ub一行放不下，需要分为padLeft、padRight和body三部分进行处理
        if (outHeight == 1 && copyTimesOneLine != 1) {
            calCount = CAL_COUNT;
            for (size_t i = 0; i < height; i++) {
                gmXOffset1 = i * width + loop * batchStride + ncOffset * batchStride;
                gmXOffset2 = (width - calCount) + i * width + loop * batchStride + ncOffset * batchStride;
                set_flag(PIPE_S, PIPE_MTE2, eventId0);
                wait_flag(PIPE_S, PIPE_MTE2, eventId0);
                CopyGm2UB(gmXOffset1, gmXOffset2, calCount);
                if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                    ComputeHGradBF16(CONST_VALUE_2 * calCount);  // 计算结果存在floatQueue
                } else {
                    ComputeHGrad(CONST_VALUE_2 * calCount);  //计算结果存在yOutQueue
                }
            }
            // workspace上的偏移量
            workspaceOffset1 = loop * CONST_VALUE_2 * calCount + ncOffset * CONST_VALUE_2 * calCount;
            // 左右两侧分别进行累加计算到edge
            if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                ComputeWGradBF16(calCount);
            } else {
                ComputeWGrad(calCount);
            }
            // 一共64列搬运到workspace上
            CopyOut2Workspace(workspaceOffset1, CONST_VALUE_2 * calCount);
            // 计算需要搬出的workspace偏移
            workspaceOffset2 = padLeft + loop * CONST_VALUE_2 * calCount + ncOffset * CONST_VALUE_2 * calCount;
            workspaceOffset3 = calCount + loop * CONST_VALUE_2 * calCount + ncOffset * CONST_VALUE_2 * calCount;
            set_flag(PIPE_S, PIPE_MTE2, eventId0);
            // 计算搬出到gm上的偏移
            gmYOffset1 = loop * outBatchStride + ncOffset * outBatchStride;
            gmYOffset2 = (outWidth - calCount + padRight) + loop * outBatchStride + ncOffset * outBatchStride;
            set_flag(PIPE_S, PIPE_MTE3, eventId2);
            // 左侧workspace -> ub -> gm
            set_flag(PIPE_MTE3, PIPE_MTE2, eventId1);
            CopyWorkspace2Out(workspaceOffset2, gmYOffset1, calCount - padLeft);
            wait_flag(PIPE_MTE3, PIPE_MTE2, eventId1);
            // 右侧workspace -> ub -> gm
            set_flag(PIPE_MTE3, PIPE_MTE2, eventId1);
            CopyWorkspace2Out(workspaceOffset3, gmYOffset2, calCount - padRight);
            wait_flag(PIPE_S, PIPE_MTE2, eventId0);
            wait_flag(PIPE_S, PIPE_MTE3, eventId2);
            wait_flag(PIPE_MTE3, PIPE_MTE2, eventId1);
            
            // 处理中间body，搬入ub累加再搬出到gm
            int64_t dataCountBody = width - CONST_VALUE_2 * calCount;
            int64_t copyTimesBody = CeilDiv(dataCountBody, ubFactorElement);
            for (size_t time = 0; time < copyTimesBody; time++) {
                copyCount = ubFactorElement;
                if (time == copyTimesBody - 1) {
                    copyCount = dataCountBody - (copyTimesBody - 1) * ubFactorElement;  // 尾块搬运数量
                }
                for (size_t i = 0; i < height; i++) {
                    gmXOffset3 = calCount + i * width + time * ubFactorElement + 
                                 loop * batchStride + ncOffset * batchStride;
                    set_flag(PIPE_S, PIPE_MTE2, eventId0);
                    wait_flag(PIPE_S, PIPE_MTE2, eventId0);
                    CopyGm2UBWhole(gmXOffset3, copyCount);
                    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                        ComputeHGradBF16(copyCount);  // 计算结果存在floatQueue
                    } else {
                        ComputeHGrad(copyCount);  //计算结果存在yOutQueue
                    }
                }
                if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                    FloatCast2BF16(copyCount);
                }
                // 输出body的起始位置
                gmYOffset3 = (calCount - padLeft) + time * ubFactorElement + 
                             loop * outBatchStride + ncOffset * outBatchStride;
                set_flag(PIPE_S, PIPE_MTE3, eventId2);
                wait_flag(PIPE_S, PIPE_MTE3, eventId2);
                CopyOut2Gm(gmYOffset3, copyCount);
            }
        }
    }
}
#endif  // _PAD_V3_GRAD_REPLICATE_H_W_