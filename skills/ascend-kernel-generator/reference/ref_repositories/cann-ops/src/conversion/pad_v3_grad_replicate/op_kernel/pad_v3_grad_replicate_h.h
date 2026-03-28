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
 * \file pad_v3_grad_replicate_h.h
 * \brief
 */
#ifndef _PAD_V3_GRAD_REPLICATE_H_
#define _PAD_V3_GRAD_REPLICATE_H_

#include "kernel_operator.h"
#include "pad_v3_grad_replicate_base.h"

using namespace AscendC;

template <typename T>
class PadV3GradReplicateH {
public:
    __aicore__ inline PadV3GradReplicateH() {};
    __aicore__ inline void Init(const PadV3GradReplicateTilingData &__restrict tilingData, 
                                GM_ADDR x, GM_ADDR padding, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void InitBuffer(TPipe *inputPipe);
    __aicore__ inline void CopyFromGm2UB(const int64_t offset, const int64_t copyCount);
    __aicore__ inline void CopyOut2Gm(const int64_t offset, const int64_t calCount);
    __aicore__ inline void CopyInAndOut2Gm(const int64_t offset1, const int64_t offset2,
                                           const int64_t calCount, const int32_t blkIdx);
    __aicore__ inline void ComputeHGrad(const int64_t calCount);
    __aicore__ inline void ComputeHGradBF16(const int64_t calCount);
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

    GlobalTensor<T> mGmX;
    GlobalTensor<T> mGmY;
};

template <typename T>
__aicore__ inline void PadV3GradReplicateH<T>::Init(const PadV3GradReplicateTilingData &__restrict tilingData,
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
    outBatchStride = outHeight * width;
    blockIdx = GetBlockIdx();
    perBlockCount = BLOCK_BYTES / sizeof(T);
    eventId0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    eventId1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));

    mGmX.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
    mGmY.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));
}

// init used buffer
template <typename T>
__aicore__ inline void PadV3GradReplicateH<T>::InitBuffer(TPipe *inputPipe) {
    pipe = inputPipe;
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        pipe->InitBuffer(xInQueue, BUFFER_NUM, ubFactorElement * sizeof(T) * CONST_VALUE_2);
        pipe->InitBuffer(yOutQueue, BUFFER_NUM, ubFactorElement * sizeof(T) * CONST_VALUE_2);
        pipe->InitBuffer(floatQueue, BUFFER_NUM, ubFactorElement * sizeof(float));
        pipe->InitBuffer(floatCastResBuf, ubFactorElement * sizeof(float));
    } else {
        pipe->InitBuffer(xInQueue, BUFFER_NUM, ubFactorElement * sizeof(T) * CONST_VALUE_2);
        pipe->InitBuffer(yOutQueue, BUFFER_NUM, ubFactorElement * sizeof(T) * CONST_VALUE_2);
    }
}

template <typename T>
__aicore__ inline void PadV3GradReplicateH<T>::CopyFromGm2UB(const int64_t offset, const int64_t copyCount) {
    LocalTensor<T> dataLocal = xInQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(copyCount, perBlockCount) - copyCount), (T)0};

    DataCopyPad(dataLocal[0], mGmX[offset], copyParams, padParams);
    pipe_barrier(PIPE_MTE2);
    xInQueue.EnQue(dataLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateH<T>::CopyOut2Gm(const int64_t offset, const int64_t calCount) {
    LocalTensor<T> dstLocal = yOutQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(mGmY[offset], dstLocal, copyParams);
    yOutQueue.FreeTensor(dstLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateH<T>::CopyInAndOut2Gm(const int64_t offset1, const int64_t offset2,
                                                               const int64_t calCount, const int32_t blkIdx) {
    LocalTensor<T> dstLocal = yOutQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(calCount, perBlockCount) - calCount), (T)0};
    wait_flag(PIPE_S, PIPE_MTE2, eventId0);
    DataCopyPad(dstLocal[blkIdx * ubFactorElement], mGmX[offset1], copyParams, padParams);
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    set_flag(PIPE_MTE2, PIPE_MTE3, eventID);
    wait_flag(PIPE_MTE2, PIPE_MTE3, eventID);
    DataCopyPad(mGmY[offset2], dstLocal[blkIdx * ubFactorElement], copyParams);
    yOutQueue.FreeTensor(dstLocal);
    set_flag(PIPE_S, PIPE_MTE2, eventId0);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateH<T>::ComputeHGrad(const int64_t calCount) {
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
    Add(yLocal, yLocal, xLocal[0], calCount);
    yOutQueue.EnQue(yLocal);
    xInQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateH<T>::ComputeHGradBF16(const int64_t calCount) {
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
__aicore__ inline void PadV3GradReplicateH<T>::FloatCast2BF16(const int64_t calCount) {
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    LocalTensor<float> floatLocal = floatQueue.DeQue<float>();
    Cast(yLocal, floatLocal, RoundMode::CAST_RINT, calCount);
    yOutQueue.EnQue(yLocal);
    floatQueue.FreeTensor(floatLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateH<T>::Process() {
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
    // 对齐场景下，ubFactorElement应为16的倍数
    uint32_t copyTimesOneLine = CeilDiv(width, ubFactorElement);  // ubFactorElement：一行元素个数

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

    for (size_t loop = 0; loop < loopNC; loop++) {
        int64_t calCount = ubFactorElement;
        for (size_t time = 0; time < copyTimesOneLine; time++) {
            if (time == copyTimesOneLine - 1) {
                calCount = width - (copyTimesOneLine - 1) * ubFactorElement;  // 尾块搬运数量
            }
            // 场景1：输出shape的H维度为1，padTop和padBottom累加的边缘行重叠，梯度要全部累加到outHeight上
            if (outHeight == 1) {
                for (size_t i = 0; i < height; i++) {
                    gmXOffset = i * width + time * ubFactorElement + loop * batchStride + ncOffset * batchStride;
                    set_flag(PIPE_S, PIPE_MTE2, eventId0);
                    wait_flag(PIPE_S, PIPE_MTE2, eventId0);
                    CopyFromGm2UB(gmXOffset, calCount);
                    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                        ComputeHGradBF16(calCount);
                    } else {
                        ComputeHGrad(calCount);
                    }
                }
                if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                    FloatCast2BF16(calCount);
                }
                gmYOffset = time * ubFactorElement + loop * outBatchStride + ncOffset * outBatchStride;
                set_flag(PIPE_S, PIPE_MTE3, eventId1);
                wait_flag(PIPE_S, PIPE_MTE3, eventId1);
                CopyOut2Gm(gmYOffset, calCount);
                continue;
            }

            // 场景2：输出shape的H维度不为1，即padTop和padBottom累加的边缘行不重叠，分三部分处理：padTop、padBottom和body
            // 处理padTop,梯度累加到边缘行
            for (size_t i = 0; i <= padTop; i++) {
                // 搬一行，padTop行一直到边缘行，梯度累加
                gmXOffset1 = i * width + time * ubFactorElement + loop * batchStride + ncOffset * batchStride;
                set_flag(PIPE_S, PIPE_MTE2, eventId0);
                wait_flag(PIPE_S, PIPE_MTE2, eventId0);
                CopyFromGm2UB(gmXOffset1, calCount);
                if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                    ComputeHGradBF16(calCount);
                } else {
                    ComputeHGrad(calCount);
                }
            }
            if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                FloatCast2BF16(calCount);
            }
            // padTop累加完成，输出到边缘首行
            gmYOffset1 = time * ubFactorElement + loop * outBatchStride + ncOffset * outBatchStride;
            set_flag(PIPE_S, PIPE_MTE3, eventId1);
            wait_flag(PIPE_S, PIPE_MTE3, eventId1);
            CopyOut2Gm(gmYOffset1, calCount);

            // 处理padBottom，梯度累加到边缘行
            for (size_t i = 0; i <= padBottom; i++) {
                // 搬一行，padBottom行一直到边缘行，梯度累加
                gmXOffset2 = (height - 1 - i) * width + time * ubFactorElement
                            + loop * batchStride + ncOffset * batchStride;
                set_flag(PIPE_S, PIPE_MTE2, eventId0);
                wait_flag(PIPE_S, PIPE_MTE2, eventId0);
                CopyFromGm2UB(gmXOffset2, calCount);
                if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                    ComputeHGradBF16(calCount);
                } else {
                    ComputeHGrad(calCount);
                }
            }
            if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                FloatCast2BF16(calCount);
            }
            // padBottom累加完成，输出到边缘尾行
            gmYOffset2 = (outHeight - 1) * width + time * ubFactorElement +
                        loop * outBatchStride + ncOffset * outBatchStride;
            set_flag(PIPE_S, PIPE_MTE3, eventId1);
            wait_flag(PIPE_S, PIPE_MTE3, eventId1);
            CopyOut2Gm(gmYOffset2, calCount);

            // 处理中间body，搬入ub再搬出到gm即可，不做计算
            for (size_t i = padTop + 1; i < height - 1 - padBottom; i++) {
                // 输入body的起始位置
                gmXOffset3 = i * width + time * ubFactorElement + loop * batchStride + ncOffset * batchStride;
                // 输出body的起始位置
                gmYOffset3 = (i - padTop) * width + time * ubFactorElement +
                            loop * outBatchStride + ncOffset * outBatchStride;
                pipe_barrier(PIPE_ALL);
                set_flag(PIPE_S, PIPE_MTE2, eventId0);
                CopyInAndOut2Gm(gmXOffset3, gmYOffset3, calCount, 0);
                wait_flag(PIPE_S, PIPE_MTE2, eventId0);
            }
        }
    }
}
#endif  // _PAD_V3_GRAD_REPLICATE_H_