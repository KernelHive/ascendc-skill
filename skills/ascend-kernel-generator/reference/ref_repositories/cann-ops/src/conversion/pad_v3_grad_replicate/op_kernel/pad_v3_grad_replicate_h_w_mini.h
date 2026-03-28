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
 * \file pad_v3_grad_replicate_mini.h
 * \brief
 */
#ifndef _PAD_V3_GRAD_REPLICATE_H_W_MINI_
#define _PAD_V3_GRAD_REPLICATE_H_W_MINI_

#include "kernel_operator.h"
#include "pad_v3_grad_replicate_base.h"

using namespace AscendC;

template <typename T>
class PadV3GradReplicateHWMini {
public:
    __aicore__ inline PadV3GradReplicateHWMini() {};
    __aicore__ inline void Init(const PadV3GradReplicateTilingData &__restrict tilingData, 
                                GM_ADDR x, GM_ADDR padding, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void InitBuffer(TPipe *inputPipe);
    __aicore__ inline void CopyIn(const int32_t loop);
    __aicore__ inline void Compute();
    __aicore__ inline void ComputeBF16();
    __aicore__ inline void CopyOut(const int32_t loop);
    __aicore__ inline void Process();

private:
    TPipe *pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yOutQueue;
    TBuf<TPosition::VECCALC> transposeBuf;
    TBuf<TPosition::VECCALC> floatCastResBuf;
    LocalTensor<T> transposeData;
    LocalTensor<float> floatTransposeData;
    LocalTensor<float> floatTenosr;

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

    GlobalTensor<T> mGmX;
    GlobalTensor<T> mGmY;
    GlobalTensor<T> mGmWorkspace;
};

template <typename T>
__aicore__ inline void PadV3GradReplicateHWMini<T>::Init(const PadV3GradReplicateTilingData &__restrict tilingData,
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
__aicore__ inline void PadV3GradReplicateHWMini<T>::InitBuffer(TPipe *inputPipe) {
    pipe = inputPipe;
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        pipe->InitBuffer(xInQueue, 1, ubFactorElement * sizeof(T) * MINI_SHAPE_MAX_ROWS);
        pipe->InitBuffer(yOutQueue, 1, ubFactorElement * sizeof(T) * MINI_SHAPE_MAX_ROWS);
        pipe->InitBuffer(transposeBuf, ubFactorElement * sizeof(float) * MINI_SHAPE_MAX_ROWS);
        pipe->InitBuffer(floatCastResBuf, ubFactorElement * sizeof(float) * MINI_SHAPE_MAX_ROWS);
    } else {
        pipe->InitBuffer(xInQueue, 1, ubFactorElement * sizeof(T) * MINI_SHAPE_MAX_ROWS);
        pipe->InitBuffer(yOutQueue, 1, ubFactorElement * sizeof(T) * MINI_SHAPE_MAX_ROWS);
        pipe->InitBuffer(transposeBuf, ubFactorElement * sizeof(T) * MINI_SHAPE_MAX_ROWS);
    }
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWMini<T>::CopyIn(const int32_t loop) {
    DataCopyExtParams copyParams{1, (uint32_t)(width * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(width, perBlockCount) - width), (T)0};
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    int64_t offset = 0;
    for (size_t i = 0; i < height; i++) {
        offset = i * width + loop * batchStride + ncOffset * batchStride;
        DataCopyPad(xLocal[i * ubFactorElement], mGmX[offset], copyParams, padParams);
    }
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWMini<T>::Compute() {
    uint64_t srcLocalList0[16];
    uint64_t dstLocalList0[16];
    uint64_t srcLocalList1[16];
    uint64_t dstLocalList1[16];
    size_t fp16TransTimes = CeilDiv(alignWidth, HALF_BLOCK_NUM);
    size_t fp32TransTimes = CeilDiv(alignWidth, FLOAT_BLOCK_NUM);
    size_t fp16TransBackTimes = CeilDiv(MINI_SHAPE_MAX_ROWS, HALF_BLOCK_NUM);
    size_t fp32TransBackTimes = CeilDiv(MINI_SHAPE_MAX_ROWS, FLOAT_BLOCK_NUM);
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = 1;
    transDataParams.dstRepStride = 0;
    transDataParams.srcRepStride = 0;

    // compute h grad
    for (size_t i = 0; i < padTop; i++) {
        Add(xLocal[padTop * ubFactorElement], xLocal[i * ubFactorElement],
            xLocal[padTop * ubFactorElement], ubFactorElement);
    }
    for (size_t i = 0; i < padBottom; i++) {
        Add(xLocal[(height - 1 - padBottom) * ubFactorElement], xLocal[(height - 1 - i) * ubFactorElement],
            xLocal[(height - 1 - padBottom) * ubFactorElement], ubFactorElement);
    }
    DataCopy(xLocal, xLocal[padTop * ubFactorElement], outHeight * ubFactorElement);
    if constexpr (AscendC::IsSameType<T, half>::value) {
        for (size_t time = 0; time < fp16TransTimes; time++) {
            for (int i = 0; i < HALF_BLOCK_NUM; i++) {
                srcLocalList0[i] = (uint64_t)(xLocal[i * ubFactorElement + time * HALF_BLOCK_NUM].GetPhyAddr());
                dstLocalList0[i] =
                    (uint64_t)(transposeData[i * MINI_SHAPE_MAX_ROWS + time * MINI_SHAPE_MAX_ROWS * HALF_BLOCK_NUM]
                                   .GetPhyAddr());
            }
            transDataParams.repeatTimes = MINI_SHAPE_MAX_ROWS / TRANSDATA_BASE_H;
            transDataParams.srcRepStride = ubFactorElement;
            transDataParams.dstRepStride = 1;
            TransDataTo5HD<T>(dstLocalList0, srcLocalList0, transDataParams);
        }
        // compute w grad
        for (size_t i = 0; i < padLeft; i++) {
            Add(transposeData[padLeft * MINI_SHAPE_MAX_ROWS], transposeData[i * MINI_SHAPE_MAX_ROWS],
                transposeData[padLeft * MINI_SHAPE_MAX_ROWS], MINI_SHAPE_MAX_ROWS);
        }
        for (size_t i = 0; i < padRight; i++) {
            Add(transposeData[(width - 1 - padRight) * MINI_SHAPE_MAX_ROWS],
                transposeData[(width - 1 - i) * MINI_SHAPE_MAX_ROWS],
                transposeData[(width - 1 - padRight) * MINI_SHAPE_MAX_ROWS], MINI_SHAPE_MAX_ROWS);
        }
        DataCopy(transposeData, transposeData[padLeft * MINI_SHAPE_MAX_ROWS],
                 (width - padLeft - padRight) * MINI_SHAPE_MAX_ROWS);
        for (size_t time = 0; time < fp16TransBackTimes; time++) {
            for (int i = 0; i < HALF_BLOCK_NUM; i++) {
                srcLocalList1[i] = (uint64_t)(transposeData[i * MINI_SHAPE_MAX_ROWS + time * HALF_BLOCK_NUM]
                                                  .GetPhyAddr());
                dstLocalList1[i] =
                    (uint64_t)(xLocal[i * ubFactorElement + time * ubFactorElement * HALF_BLOCK_NUM]
                                   .GetPhyAddr());
            }
            transDataParams.repeatTimes = ubFactorElement / TRANSDATA_BASE_H;
            transDataParams.srcRepStride = MINI_SHAPE_MAX_ROWS;
            transDataParams.dstRepStride = 1;
            TransDataTo5HD<T>(dstLocalList1, srcLocalList1, transDataParams);
        }
        DataCopy(yLocal, xLocal, MINI_SHAPE_MAX_ROWS * ubFactorElement);
        xInQueue.FreeTensor(xLocal);
        yOutQueue.EnQue(yLocal);
    } else {
        for (size_t time = 0; time < fp32TransTimes; time++) {
            for (int i = 0; i < HALF_BLOCK_NUM; i++) {
                srcLocalList0[i] = (uint64_t)(xLocal[i * ubFactorElement + time * FLOAT_BLOCK_NUM].GetPhyAddr());
            }
            for (int i = 0; i < FLOAT_BLOCK_NUM; i++) {
                dstLocalList0[CONST_VALUE_2 * i] =
                    (uint64_t)(transposeData[i * MINI_SHAPE_MAX_ROWS + time * MINI_SHAPE_MAX_ROWS * FLOAT_BLOCK_NUM]
                                   .GetPhyAddr());
                dstLocalList0[CONST_VALUE_2 * i + 1] =
                    (uint64_t)(transposeData[i * MINI_SHAPE_MAX_ROWS + time * MINI_SHAPE_MAX_ROWS * FLOAT_BLOCK_NUM +
                                             FLOAT_BLOCK_NUM]
                                   .GetPhyAddr());
            }
            transDataParams.repeatTimes = MINI_SHAPE_MAX_ROWS / TRANSDATA_BASE_H;
            transDataParams.srcRepStride = CONST_VALUE_2 * ubFactorElement;
            transDataParams.dstRepStride = CONST_VALUE_2;
            TransDataTo5HD<T>(dstLocalList0, srcLocalList0, transDataParams);
        }
        for (size_t i = 0; i < padLeft; i++) {
            Add(transposeData[padLeft * MINI_SHAPE_MAX_ROWS], transposeData[i * MINI_SHAPE_MAX_ROWS],
                transposeData[padLeft * MINI_SHAPE_MAX_ROWS], MINI_SHAPE_MAX_ROWS);
        }
        for (size_t i = 0; i < padRight; i++) {
            Add(transposeData[(width - 1 - padRight) * MINI_SHAPE_MAX_ROWS],
                transposeData[(width - 1 - i) * MINI_SHAPE_MAX_ROWS],
                transposeData[(width - 1 - padRight) * MINI_SHAPE_MAX_ROWS], MINI_SHAPE_MAX_ROWS);
        }
        DataCopy(transposeData, transposeData[padLeft * MINI_SHAPE_MAX_ROWS],
                 (width - padLeft - padRight) * MINI_SHAPE_MAX_ROWS);
        for (size_t time = 0; time < fp32TransBackTimes; time++) {
            for (int i = 0; i < HALF_BLOCK_NUM; i++) {
                srcLocalList1[i] =
                    (uint64_t)(transposeData[MINI_SHAPE_MAX_ROWS * i + time * FLOAT_BLOCK_NUM].GetPhyAddr());
            }
            for (int i = 0; i < FLOAT_BLOCK_NUM; i++) {
                dstLocalList1[CONST_VALUE_2 * i] =
                    (uint64_t)(xLocal[i * ubFactorElement + time * ubFactorElement * FLOAT_BLOCK_NUM]
                                   .GetPhyAddr());  // 每行首地址
                dstLocalList1[CONST_VALUE_2 * i + 1] =
                    (uint64_t)(xLocal[i * ubFactorElement + time * ubFactorElement * FLOAT_BLOCK_NUM +
                                      FLOAT_BLOCK_NUM]
                                   .GetPhyAddr());  // 每行首地址
            }
            transDataParams.repeatTimes = ubFactorElement / TRANSDATA_BASE_H;
            transDataParams.srcRepStride = CONST_VALUE_2 * MINI_SHAPE_MAX_ROWS;
            transDataParams.dstRepStride = CONST_VALUE_2;
            TransDataTo5HD<T>(dstLocalList1, srcLocalList1, transDataParams);
        }
        DataCopy(yLocal, xLocal, MINI_SHAPE_MAX_ROWS * ubFactorElement);
        xInQueue.FreeTensor(xLocal);
        yOutQueue.EnQue(yLocal);
    }
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWMini<T>::ComputeBF16() {
    uint64_t srcLocalList0[16];
    uint64_t dstLocalList0[16];
    uint64_t srcLocalList1[16];
    uint64_t dstLocalList1[16];
    size_t fp32TransTimes = CeilDiv(alignWidth, FLOAT_BLOCK_NUM);
    size_t fp32TransBackTimes = CeilDiv(MINI_SHAPE_MAX_ROWS, FLOAT_BLOCK_NUM);
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = 1;
    transDataParams.dstRepStride = 0;
    transDataParams.srcRepStride = 0;
    Cast(floatTenosr, xLocal, RoundMode::CAST_NONE, ubFactorElement * MINI_SHAPE_MAX_ROWS);
    // compute h grad
    for (size_t i = 0; i < padTop; i++) {
        Add(floatTenosr[padTop * ubFactorElement], floatTenosr[i * ubFactorElement],
            floatTenosr[padTop * ubFactorElement], ubFactorElement);
    }
    for (size_t i = 0; i < padBottom; i++) {
        Add(floatTenosr[(height - 1 - padBottom) * ubFactorElement],
            floatTenosr[(height - 1 - i) * ubFactorElement],
            floatTenosr[(height - 1 - padBottom) * ubFactorElement], ubFactorElement);
    }
    DataCopy(floatTenosr, floatTenosr[padTop * ubFactorElement], outHeight * ubFactorElement);

    for (size_t time = 0; time < fp32TransTimes; time++) {
        for (int i = 0; i < HALF_BLOCK_NUM; i++) {
            srcLocalList0[i] =
                (uint64_t)(floatTenosr[i * ubFactorElement + time * FLOAT_BLOCK_NUM].GetPhyAddr());
        }
        for (int i = 0; i < FLOAT_BLOCK_NUM; i++) {
            dstLocalList0[CONST_VALUE_2 * i] =
                (uint64_t)(floatTransposeData[i * MINI_SHAPE_MAX_ROWS + time * MINI_SHAPE_MAX_ROWS * FLOAT_BLOCK_NUM]
                               .GetPhyAddr());  // 每行首地址
            dstLocalList0[CONST_VALUE_2 * i + 1] =
                (uint64_t)(floatTransposeData[i * MINI_SHAPE_MAX_ROWS + time * MINI_SHAPE_MAX_ROWS * FLOAT_BLOCK_NUM +
                                              FLOAT_BLOCK_NUM]
                               .GetPhyAddr());  // 每行首地址
        }
        transDataParams.repeatTimes = MINI_SHAPE_MAX_ROWS / TRANSDATA_BASE_H;
        transDataParams.srcRepStride = CONST_VALUE_2 * ubFactorElement;
        transDataParams.dstRepStride = CONST_VALUE_2;
        TransDataTo5HD<float>(dstLocalList0, srcLocalList0, transDataParams);
    }
    for (size_t i = 0; i < padLeft; i++) {
        Add(floatTransposeData[padLeft * MINI_SHAPE_MAX_ROWS], floatTransposeData[i * MINI_SHAPE_MAX_ROWS],
            floatTransposeData[padLeft * MINI_SHAPE_MAX_ROWS], MINI_SHAPE_MAX_ROWS);
    }
    for (size_t i = 0; i < padRight; i++) {
        Add(floatTransposeData[(width - 1 - padRight) * MINI_SHAPE_MAX_ROWS],
            floatTransposeData[(width - 1 - i) * MINI_SHAPE_MAX_ROWS],
            floatTransposeData[(width - 1 - padRight) * MINI_SHAPE_MAX_ROWS], MINI_SHAPE_MAX_ROWS);
    }
    DataCopy(floatTransposeData, floatTransposeData[padLeft * MINI_SHAPE_MAX_ROWS],
             (width - padLeft - padRight) * MINI_SHAPE_MAX_ROWS);

    for (size_t time = 0; time < fp32TransBackTimes; time++) {
        for (int i = 0; i < HALF_BLOCK_NUM; i++) {
            srcLocalList1[i] =
                (uint64_t)(floatTransposeData[MINI_SHAPE_MAX_ROWS * i + time * FLOAT_BLOCK_NUM].GetPhyAddr());
        }
        for (int i = 0; i < FLOAT_BLOCK_NUM; i++) {
            dstLocalList1[CONST_VALUE_2 * i] =
                (uint64_t)(floatTenosr[i * ubFactorElement + time * ubFactorElement * FLOAT_BLOCK_NUM]
                               .GetPhyAddr());
            dstLocalList1[CONST_VALUE_2 * i + 1] =
                (uint64_t)(floatTenosr[i * ubFactorElement + time * ubFactorElement * FLOAT_BLOCK_NUM +
                                       FLOAT_BLOCK_NUM]
                               .GetPhyAddr());
        }
        transDataParams.repeatTimes = ubFactorElement / TRANSDATA_BASE_H;
        transDataParams.srcRepStride = CONST_VALUE_2 * MINI_SHAPE_MAX_ROWS;
        transDataParams.dstRepStride = CONST_VALUE_2;
        TransDataTo5HD<float>(dstLocalList1, srcLocalList1, transDataParams);
    }
    Cast(yLocal, floatTenosr, RoundMode::CAST_RINT, ubFactorElement * MINI_SHAPE_MAX_ROWS);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWMini<T>::CopyOut(const int32_t loop) {
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(outWidth * sizeof(T)), 0, 0, 0};
    int64_t offset = 0;
    for (size_t i = 0; i < outHeight; i++) {
        offset = i * outWidth + loop * outBatchStride + ncOffset * outBatchStride;
        DataCopyPad(mGmY[offset], yLocal[i * ubFactorElement], copyParams);
    }
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWMini<T>::Process() {
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        floatTransposeData = transposeBuf.Get<float>();
        floatTenosr = floatCastResBuf.Get<float>();
    } else {
        transposeData = transposeBuf.Get<T>();
    }
    for (size_t loop = 0; loop < loopNC; loop++) {
        CopyIn(loop);
        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
            ComputeBF16();
        } else {
            Compute();
        }
        CopyOut(loop);
    }
}
#endif  // _PAD_V3_GRAD_REPLICATE_H_W_MINI_