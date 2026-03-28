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
 * \file reverse_sequence_base.h
 * \brief
 */
#ifndef REVERSE_SEQUENCE_BASE_H
#define REVERSE_SEQUENCE_BASE_H

#include "kernel_operator.h"
#include "platform.h"

namespace ReverseSequence {
using namespace AscendC;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t BYTE_BLOCK = 32;

template <typename T>
class ReverseSequenceBase {
public:
    __aicore__ inline ReverseSequenceBase(const ReverseSequenceTilingData* tilingDataPtr) {
        batchDimValue_ = tilingDataPtr->batchDimValue;
        seqDimValue_ = tilingDataPtr->seqDimValue;
        xDtypeSize_ = tilingDataPtr->xDtypeSize;
        batchSize_ = tilingDataPtr->batchSize;
        seqSize_ = tilingDataPtr->seqSize;
        cSize_ = tilingDataPtr->cSize;
        maxProcCount_ = tilingDataPtr->maxProcCount;
        loopTimePerCore_ = tilingDataPtr->loopTimePerCore;
        tailCoreNum_ = tilingDataPtr->tailCoreNum;
        innerLoopTime_ = tilingDataPtr->innerLoopTime;
        innerTailCount_ = tilingDataPtr->innerTailCount;
    };

    template <typename CLS_NAME, void (CLS_NAME::*funSingleReverse)(
                                     const int64_t batchIndex, const int64_t seqGroupIndex, const int64_t reverseCount)>
    __aicore__ inline void Process(CLS_NAME* objPtr);

protected:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b) {
        T1 bTemp(b);
        return bTemp == 0 ? a : (a + bTemp - 1) / bTemp;
    };

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilAlign(T1 a, T2 b) {
        T1 bTemp(b);
        return bTemp == 0 ? a : CeilDiv(a, bTemp) * bTemp;
    };

    __aicore__ inline void BaseInit(GM_ADDR seq_lengths, GM_ADDR workspace) {
        blockIdx_ = GetBlockIdx();
        perBlockCount_ = BYTE_BLOCK / xDtypeSize_;
        seqLengthsGM_.SetGlobalBuffer((__gm__ T*)seq_lengths);
        pipe_.InitBuffer(xQue_, BUFFER_NUM, maxProcCount_ * xDtypeSize_);
        if (blockIdx_ < tailCoreNum_) {
            loopTimePerCore_ += 1;
            batchBaseIndex_ = loopTimePerCore_ * blockIdx_;
        } else {
            batchBaseIndex_ = loopTimePerCore_ * blockIdx_ + tailCoreNum_;
        }
    };

    template <typename T1>
    __aicore__ inline void CopyInData(const LocalTensor<T1>& dstUB, const GlobalTensor<T1>& srcGM,
                                      const int64_t dataCount);
    template <typename T1>
    __aicore__ inline void CopyOutData(const GlobalTensor<T1>& dstGM, const LocalTensor<T1>& srcUB,
                                       const int64_t dataCount);

    __aicore__ inline void CopyInX(const int64_t gmOffset, const int64_t dataCount);
    __aicore__ inline void CopyOutX(const int64_t gmOffset, const int64_t dataCount);
    __aicore__ inline void SingleReverseCBig(int64_t gmInOffset, int64_t gmOutOffset);
    __aicore__ inline void SingleCopyCBig(int64_t gmInOffset);

protected:
    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> xQue_;
    GlobalTensor<uint8_t> xGM_, yGM_;
    GlobalTensor<T> seqLengthsGM_;
    int64_t blockIdx_ = 0;
    int32_t perBlockCount_ = 0;
    int64_t batchBaseIndex_ = 0;

    // tiling params
    int64_t batchDimValue_ = 0;
    int64_t seqDimValue_ = 0;
    int64_t xDtypeSize_ = 0;
    int64_t batchSize_ = 0;
    int64_t seqSize_ = 0;
    int64_t cSize_ = 0;
    int64_t maxProcCount_ = 0;
    int64_t loopTimePerCore_ = 0;
    int64_t tailCoreNum_ = 0;
    int64_t innerLoopTime_ = 0;
    int64_t innerTailCount_ = 0;
};

template <typename T>
template <typename CLS_NAME, void (CLS_NAME::*funSingleReverse)(const int64_t batchIndex, const int64_t seqGroupIndex,
                                                                const int64_t reverseCount)>
__aicore__ inline void ReverseSequenceBase<T>::Process(CLS_NAME* objPtr) {
    for (int64_t i = 0; i < loopTimePerCore_; i++) {
        int64_t reverseCount = int64_t(seqLengthsGM_.GetValue((batchBaseIndex_ + i) % batchDimValue_));
        if (reverseCount <= 1) {
            reverseCount = 0;
        }
        if (reverseCount > seqDimValue_) {
            reverseCount = seqDimValue_;
        }
        for (int64_t j = 0; j < seqSize_ / seqDimValue_; j++) {
            (objPtr->*funSingleReverse)(i, j, reverseCount);
        }
    }
}

template <typename T>
template <typename T1>
__aicore__ inline void ReverseSequenceBase<T>::CopyInData(const LocalTensor<T1>& dstUB, const GlobalTensor<T1>& srcGM,
                                                          const int64_t dataCount) {
    int64_t elementsPerBlock = BYTE_BLOCK / sizeof(T1);
    if (dataCount % elementsPerBlock) {
        if constexpr (PlatformSocInfo::IsDataCopyPadSupport()) {
            DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
            copyParams.blockLen = dataCount * sizeof(T1);
            DataCopyPadExtParams<T1> padParams = {false, 0, 0, 0};
            DataCopyPad(dstUB, srcGM, copyParams, padParams);
        } else {
            int64_t floorAlignCnt = dataCount / elementsPerBlock * elementsPerBlock;
            DataCopy(dstUB, srcGM, floorAlignCnt);
            DataCopy(dstUB[floorAlignCnt], srcGM[dataCount - elementsPerBlock], elementsPerBlock);
        }
    } else {
        DataCopy(dstUB, srcGM, dataCount);
    }
}

template <typename T>
template <typename T1>
__aicore__ inline void ReverseSequenceBase<T>::CopyOutData(const GlobalTensor<T1>& dstGM, const LocalTensor<T1>& srcUB,
                                                           const int64_t dataCount) {
    int64_t elementsPerBlock = BYTE_BLOCK / sizeof(T1);
    if (dataCount % elementsPerBlock) {
        if constexpr (PlatformSocInfo::IsDataCopyPadSupport()) {
            DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
            copyParams.blockLen = dataCount * sizeof(T1);
            DataCopyPad(dstGM, srcUB, copyParams);
        } else {
            int64_t floorAlignCnt = dataCount / elementsPerBlock * elementsPerBlock;
            DataCopy(dstGM, srcUB, floorAlignCnt);
            DataCopy(dstGM[dataCount - elementsPerBlock], srcUB[floorAlignCnt], elementsPerBlock);
        }
    } else {
        DataCopy(dstGM, srcUB, dataCount);
    }
}

template <typename T>
__aicore__ inline void ReverseSequenceBase<T>::CopyInX(const int64_t gmOffset, const int64_t dataCount) {
    LocalTensor<uint8_t> xIn = xQue_.AllocTensor<uint8_t>();
    CopyInData(xIn, xGM_[gmOffset * xDtypeSize_], dataCount * xDtypeSize_);
    xQue_.EnQue(xIn);
}

template <typename T>
__aicore__ inline void ReverseSequenceBase<T>::CopyOutX(const int64_t gmOffset, const int64_t dataCount) {
    LocalTensor<uint8_t> yOut = xQue_.DeQue<uint8_t>();
    CopyOutData(yGM_[gmOffset * xDtypeSize_], yOut, dataCount * xDtypeSize_);
    xQue_.FreeTensor(yOut);
}

template <typename T>
__aicore__ inline void ReverseSequenceBase<T>::SingleReverseCBig(int64_t gmInOffset, int64_t gmOutOffset) {
    for (int64_t j = 0; j < innerLoopTime_; j++) {
        CopyInX(gmInOffset, maxProcCount_);
        CopyOutX(gmOutOffset, maxProcCount_);
        gmInOffset += maxProcCount_;
        gmOutOffset += maxProcCount_;
    }
    if (innerTailCount_) {
        CopyInX(gmInOffset, innerTailCount_);
        CopyOutX(gmOutOffset, innerTailCount_);
    }
}

template <typename T>
__aicore__ inline void ReverseSequenceBase<T>::SingleCopyCBig(int64_t gmInOffset) {
    for (int64_t j = 0; j < innerLoopTime_; j++) {
        CopyInX(gmInOffset, maxProcCount_);
        CopyOutX(gmInOffset, maxProcCount_);
        gmInOffset += maxProcCount_;
    }
    if (innerTailCount_) {
        CopyInX(gmInOffset, innerTailCount_);
        CopyOutX(gmInOffset, innerTailCount_);
    }
}

}  // namespace ReverseSequence

#endif  // REVERSE_SEQUENCE_BASE_H