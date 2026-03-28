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
 * \file pad_v3_grad_replicate_w.h
 */
#ifndef _PAD_V3_GRAD_REPLICATE_W_
#define _PAD_V3_GRAD_REPLICATE_W_

#include "kernel_operator.h"
#include "pad_v3_grad_replicate_base.h"

using namespace AscendC;

template <typename T>
class PadV3GradReplicateW {
public:
    __aicore__ inline PadV3GradReplicateW() {};
    __aicore__ inline void Init(const PadV3GradReplicateTilingData &__restrict tilingData,
                                GM_ADDR x, GM_ADDR padding, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void InitBuffer(TPipe *inputPipe);
    __aicore__ inline void CopyGm2UBWhole(const int64_t offset, const int64_t copyCount);
    __aicore__ inline void CopyGm2UB(const int64_t offset1, const int64_t offset2, const int64_t copyCount);
    __aicore__ inline void CopyWorkspace2Out(const int64_t offset1, const int64_t offset2, const int64_t copyCount);
    __aicore__ inline void CopyInAndOut2Gm(const int64_t offset1, const int64_t offset2, const int64_t calCount);
    __aicore__ inline void CopyOut2Workspace(const int64_t offset, const int64_t calCount);
    __aicore__ inline void ComputeWGrad(const int32_t calCount);
    __aicore__ inline void ComputeWGradBF16(const int32_t calCount);
    __aicore__ inline void ComputeWGradWhole(const int32_t calCount);
    __aicore__ inline void ComputeWGradWholeBF16(const int32_t calCount);
    __aicore__ inline void Process();

private:
    TPipe *pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yOutQueue;
    TBuf<TPosition::VECCALC> floatCastResBuf;
    
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
    uint32_t wCalCount = 0;
    event_t eventId0;
    event_t eventId1;
    event_t eventId2;
    event_t eventId3;

    GlobalTensor<T> mGmX;
    GlobalTensor<T> mGmY;
    GlobalTensor<T> mGmWorkspace;
};

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::Init(const PadV3GradReplicateTilingData &__restrict tilingData,
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
    wCalCount = tilingData.wCalCount;

    batchStride = width;
    outBatchStride = outWidth;
    blockIdx = GetBlockIdx();
    perBlockCount = BLOCK_BYTES / sizeof(T);
    eventId0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    eventId1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    eventId2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    eventId3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));

    mGmX.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
    mGmY.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));
    mGmWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace));
}

// init used buffer
template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::InitBuffer(TPipe *inputPipe) {
    pipe = inputPipe;
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        pipe->InitBuffer(xInQueue, BUFFER_NUM, wCalCount * sizeof(T) * CONST_VALUE_2);
        pipe->InitBuffer(yOutQueue, BUFFER_NUM, ubFactorElement * sizeof(T));
        pipe->InitBuffer(floatCastResBuf, ubFactorElement * sizeof(float));
    } else {
        pipe->InitBuffer(xInQueue, BUFFER_NUM, wCalCount * sizeof(T) * CONST_VALUE_2);
        pipe->InitBuffer(yOutQueue, BUFFER_NUM, ubFactorElement * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::CopyGm2UBWhole(const int64_t offset, const int64_t copyCount) {
    LocalTensor<T> dataLocal = xInQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(copyCount, perBlockCount) - copyCount), (T)0};

    DataCopyPad(dataLocal, mGmX[offset], copyParams, padParams);
    xInQueue.EnQue(dataLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::CopyGm2UB(const int64_t offset1, const int64_t offset2,
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
__aicore__ inline void PadV3GradReplicateW<T>::CopyWorkspace2Out(const int64_t offset1, const int64_t offset2,
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
__aicore__ inline void PadV3GradReplicateW<T>::CopyInAndOut2Gm(const int64_t offset1, const int64_t offset2,
                                                               const int64_t calCount) {
    LocalTensor<T> dstLocal = yOutQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(calCount, perBlockCount) - calCount), (T)0};
    wait_flag(PIPE_S, PIPE_MTE2, eventId0);
    DataCopyPad(dstLocal, mGmX[offset1], copyParams, padParams);
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    set_flag(PIPE_MTE2, PIPE_MTE3, eventID);
    wait_flag(PIPE_MTE2, PIPE_MTE3, eventID);
    wait_flag(PIPE_S, PIPE_MTE3, eventId2);
    DataCopyPad(mGmY[offset2], dstLocal, copyParams);
    yOutQueue.FreeTensor(dstLocal);
    set_flag(PIPE_S, PIPE_MTE2, eventId0);
    set_flag(PIPE_S, PIPE_MTE3, eventId2);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::CopyOut2Workspace(const int64_t offset, const int64_t calCount) {
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(CONST_VALUE_2 * calCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(mGmWorkspace[offset], yLocal, copyParams);  // 拷贝到workspace，注意计算偏移
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::ComputeWGrad(const int32_t calCount) {
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    T val1;
    T val2;
    T val3;
    T val4;
    T tmp1;
    T tmp2;
    for (size_t i = 0; i < padLeft; i++) {
        val1 = xLocal.GetValue(i);  // index
        val2 = xLocal.GetValue(padLeft);  // index 边缘轴
        if constexpr(AscendC::IsSameType<T, half>::value) {
            tmp1 = (T)((float)val1 + (float)val2);
            xLocal.SetValue(padLeft, tmp1);
        } else {
            xLocal.SetValue(padLeft, val1 + val2);
        }
    }
    for (size_t i = 0; i < padRight; i++) {
        val3 = xLocal.GetValue(CONST_VALUE_2 * calCount - 1 - i);  // index
        val4 = xLocal.GetValue(CONST_VALUE_2 * calCount - 1 - padRight);  // index 边缘轴
        if constexpr(AscendC::IsSameType<T, half>::value) {
            tmp2 = (T)((float)val3 + (float)val4);
            xLocal.SetValue(CONST_VALUE_2 * calCount - 1 - padRight, tmp2);
        } else {
            xLocal.SetValue(CONST_VALUE_2 * calCount - 1 - padRight, val3 + val4);
        }
    }
    DataCopy(yLocal, xLocal, CONST_VALUE_2 * calCount);
    yOutQueue.EnQue(yLocal);
    xInQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::ComputeWGradBF16(const int32_t calCount) {
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    LocalTensor<float> floatTensor = floatCastResBuf.Get<float>();
    float val1;
    float val2;
    float val3;
    float val4;
    Cast(floatTensor, xLocal, RoundMode::CAST_NONE, CONST_VALUE_2 * calCount);
    for (size_t i = 0; i < padLeft; i++) {
        val1 = floatTensor.GetValue(i);  // index
        val2 = floatTensor.GetValue(padLeft);  // index 边缘轴
        floatTensor.SetValue(padLeft, val1 + val2);
    }
    for (size_t i = 0; i < padRight; i++) {
        val3 = floatTensor.GetValue(CONST_VALUE_2 * calCount - 1 - i);  // index
        val4 = floatTensor.GetValue(CONST_VALUE_2 * calCount - 1 - padRight);  // index 边缘轴
        floatTensor.SetValue(CONST_VALUE_2 * calCount - 1 - padRight, val3 + val4);
    }
    Cast(yLocal, floatTensor, RoundMode::CAST_ROUND, CONST_VALUE_2 * calCount);
    yOutQueue.EnQue(yLocal);
    xInQueue.FreeTensor(xLocal);
    floatCastResBuf.FreeTensor(floatTensor);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::ComputeWGradWhole(const int32_t calCount) {
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    T val1;
    T val2;
    T val3;
    T val4;
    T tmp1;
    T tmp2;
    for (size_t i = 0; i < padLeft; i++) {
        val1 = xLocal.GetValue(i);  // index
        val2 = xLocal.GetValue(padLeft);  // index 边缘轴
        if constexpr(AscendC::IsSameType<T, half>::value) {
            tmp1 = (T)((float)val1 + (float)val2);
            xLocal.SetValue(padLeft, tmp1);
        } else {
            xLocal.SetValue(padLeft, val1 + val2);
        }
    }
    for (size_t i = 0; i < padRight; i++) {
        val3 = xLocal.GetValue(width - 1 - i);  // index
        val4 = xLocal.GetValue(width - 1 - padRight);  // index 边缘轴
        if constexpr(AscendC::IsSameType<T, half>::value) {
            tmp2 = (T)((float)val3 + (float)val4);
            xLocal.SetValue(width - 1 - padRight, tmp2);
        } else {
            xLocal.SetValue(width - 1 - padRight, val3 + val4);
        }
    }
    DataCopy(yLocal, xLocal, CONST_VALUE_2 * calCount);
    yOutQueue.EnQue(yLocal);
    xInQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::ComputeWGradWholeBF16(const int32_t calCount) {
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    LocalTensor<float> floatTensor = floatCastResBuf.Get<float>();
    float val1;
    float val2;
    float val3;
    float val4;
    Cast(floatTensor, xLocal, RoundMode::CAST_NONE, CONST_VALUE_2 * calCount);
    for (size_t i = 0; i < padLeft; i++) {
        val1 = floatTensor.GetValue(i);  // index
        val2 = floatTensor.GetValue(padLeft);  // index 边缘轴
        floatTensor.SetValue(padLeft, val1 + val2);
    }
    for (size_t i = 0; i < padRight; i++) {
        val3 = floatTensor.GetValue(width - 1 - i);  // index
        val4 = floatTensor.GetValue(width - 1 - padRight);  // index 边缘轴
        floatTensor.SetValue(width - 1 - padRight, val3 + val4);
    }
    Cast(yLocal, floatTensor, RoundMode::CAST_ROUND, CONST_VALUE_2 * calCount);
    yOutQueue.EnQue(yLocal);
    xInQueue.FreeTensor(xLocal);
    floatCastResBuf.FreeTensor(floatTensor);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::Process() {
    uint32_t loopNC = 0;
    int64_t ncOffset;
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
    int64_t calCount = wCalCount;
    int64_t dataCountOneLine = width - CONST_VALUE_2 * calCount;  // 中间body部分，不参与计算，复制即可
    // 对齐场景下，ubFactorElement应为16的倍数
    uint32_t copyTimesOneLine = CeilDiv(dataCountOneLine, ubFactorElement);  // ubFactorElement：一行元素个数

    if (blockIdx < tailNC) {
        loopNC = ncPerCore + 1;
        ncOffset = blockIdx * loopNC;
    } else {
        loopNC = ncPerCore;
        ncOffset = blockIdx * ncPerCore + tailNC;
    }
    // 场景1：输入shape的W维度不超过2 * wCalCount，可以完全将整行搬到ub上，进行累加计算
    if (width <= CONST_VALUE_2 * wCalCount) {
        for (size_t loop = 0; loop < loopNC; loop++) {
            gmXOffset = loop * batchStride + ncOffset * batchStride;
            workspaceOffset = blockIdx * CONST_VALUE_2 * calCount;  // workspace上的64空间
            set_flag(PIPE_S, PIPE_MTE2, eventId0);
            wait_flag(PIPE_S, PIPE_MTE2, eventId0);
            CopyGm2UBWhole(gmXOffset, calCount * CONST_VALUE_2);
            // 左右两侧分别进行累加计算到edge
            if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                ComputeWGradWholeBF16(calCount);
            } else {
                ComputeWGradWhole(calCount);
            }
            // ub一共64列搬运到workspce上
            CopyOut2Workspace(workspaceOffset, calCount);
            // 计算需要搬出的worksapce偏移，需要搬出的起始位置，即左侧边缘行
            workspaceOffsetOut = padLeft + blockIdx * CONST_VALUE_2 * calCount;
            set_flag(PIPE_S, PIPE_MTE2, eventId0);
            // 计算搬出到gm上的偏移，outWidth左侧的起始位置，index0
            gmYOffset = loop * outBatchStride + ncOffset * outBatchStride;
            set_flag(PIPE_S, PIPE_MTE3, eventId2);
            // workspace -> ub -> gm
            set_flag(PIPE_MTE3, PIPE_MTE2, eventId1);
            CopyWorkspace2Out(workspaceOffsetOut, gmYOffset, outWidth);
            wait_flag(PIPE_S, PIPE_MTE2, eventId0);
            wait_flag(PIPE_S, PIPE_MTE3, eventId2);
            wait_flag(PIPE_MTE3, PIPE_MTE2, eventId1);
        }
        return;
    }
    // 场景2：输入shape的W维度大于2 * wCalCount，需要分为padLeft、padRight和body三部分进行处理
    for (size_t loop = 0; loop < loopNC; loop++) {
        int64_t copyCount = ubFactorElement;
        gmXOffset1 = loop * batchStride + ncOffset * batchStride;  // 包含padLeft的最左侧32列起始位置
        gmXOffset2 = (width - calCount) + loop * batchStride + ncOffset * batchStride; // 包含padRight的最右侧32列起始位置
        workspaceOffset1 = blockIdx * CONST_VALUE_2 * calCount;  // workspace上的64空间
        // 左右两侧分别搬运到UB上
        set_flag(PIPE_S, PIPE_MTE2, eventId0);
        wait_flag(PIPE_S, PIPE_MTE2, eventId0);
        CopyGm2UB(gmXOffset1, gmXOffset2, calCount);
        // 左右两侧分别进行累加计算到edge
        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
            ComputeWGradBF16(calCount);
        } else {
            ComputeWGrad(calCount);
        }
        // 一共64列搬运到workspce上
        CopyOut2Workspace(workspaceOffset1, calCount);
        // 计算需要搬出的worksapce偏移
        // 左侧需要搬出的起始位置，即左侧边缘行~index31
        workspaceOffset2 = padLeft + blockIdx * CONST_VALUE_2 * calCount;
        // 右侧需要搬出的起始位置，即index32~右侧边缘行
        workspaceOffset3 = calCount + blockIdx * CONST_VALUE_2 * calCount;
        set_flag(PIPE_S, PIPE_MTE2, eventId0);
        // 计算搬出到gm上的偏移
        // outWidth左侧的起始位置，index0
        gmYOffset1 = loop * outBatchStride + ncOffset * outBatchStride;
        // outWidth右侧的起始位置，(outWidth - calCount + padRight)
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
        // 处理中间body，搬入ub再搬出到gm即可，不做计算
        for (size_t time = 0; time < copyTimesOneLine; time++) {
            if (time == copyTimesOneLine - 1) {
                copyCount = dataCountOneLine - (copyTimesOneLine - 1) * ubFactorElement;  // 尾块搬运数量
            }
            // 输入body的起始位置
            gmXOffset3 = calCount + time * ubFactorElement + loop * batchStride + ncOffset * batchStride;
            set_flag(PIPE_S, PIPE_MTE2, eventId0);
            // 输出body的起始位置
            gmYOffset3 = (calCount - padLeft) + time * ubFactorElement +
                          loop * outBatchStride + ncOffset * outBatchStride;
            pipe_barrier(PIPE_ALL);
            set_flag(PIPE_S, PIPE_MTE3, eventId2);
            CopyInAndOut2Gm(gmXOffset3, gmYOffset3, copyCount);
            wait_flag(PIPE_S, PIPE_MTE2, eventId0);
            wait_flag(PIPE_S, PIPE_MTE3, eventId2);
        }
    }
}
#endif  // _PAD_V3_GRAD_REPLICATE_W_